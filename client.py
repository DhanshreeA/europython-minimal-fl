import json
from urllib.parse import urlparse

from kafka import KafkaProducer
import socketio
from tensorflow.keras.models import model_from_json

from modeling_utils import _load_data, encode, decode


class Node:
    def __init__(self, address, partition, client, epochs=5):
        self.client = client
        self.server = address
        self.sio = socketio.Client()
        self.register_handles()

        train, test = _load_data(partition=partition)
        self.train = train.batch(4)
        self.test = test.batch(4)
        self.model = None

        self.epochs = epochs
        self.batch_size = 4

    def connect(self):
        self.sio.connect(url=self.server)

    def register_handles(self):
        self.sio.on("connection_received", self.connection_received)
        self.sio.on("start_training", self.start_training)

    def connection_received(self):
        print(f"Server at {self.server} returned success")

    def start_training(self, _model):
        self.model = model_from_json(_model["model"])
        self.model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
        print(self.model.summary())
        print("Starting training")
        self.model.set_weights(
            decode(_model["weights"])
        )
        loss = self.fit(self.model)
        self.send_updates(loss=loss)  # fix

    def fit(self, model):
        history = model.fit(
            self.train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=self.test,
        )
        return history.history["loss"][-1]

    def send_updates(self, loss):
        encoded_model_layers = [
            encode(lyr) for lyr in [layer.get_weights() for layer in self.model.layers]
        ]
        kafka_producer = KafkaProducer(
            bootstrap_servers=["0.0.0.0:9092"],
            value_serializer=lambda m: json.dumps(m).encode("ascii"),
        )
        # self.logger.debug (encoded_model_layers)
        for idx, layer in enumerate(encoded_model_layers):
            kafka_producer.send(
                "updates",
                key=self.client.encode("utf-8"),
                value={
                    "layer_id": idx,
                    "model_layer": layer,
                },
            )
        self.sio.emit("fl_update", data={"metrics": {"loss": loss}})

    def disconnect(self):
        return

    def end_session(self, data):
        model_weights = decode(data["model"])
        self.model.set_weights(model_weights)


if __name__ == "__main__":
    node = Node("http://0.0.0.0:5000", partition=2, client="client2")
    node.connect()
