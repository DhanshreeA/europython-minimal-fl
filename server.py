import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import reduce
import json
import os
from typing import Dict, List
from urllib.parse import urlparse

from aiohttp import web
from kafka import KafkaConsumer, TopicPartition
import numpy as np
import socketio


from modeling_utils import load_model_and_data, encode, decode


class Server:
    def __init__(self, min_nodes: int = 2, rounds: int = 3):
        self.sio = socketio.AsyncServer(async_mode="aiohttp")
        self.app = web.Application()
        self.sio.attach(self.app)
        self.register_handles()

        self.connected_nodes = list()
        self.pending_nodes = list()
        self.min_nodes = min_nodes
        self.training_room = "training_room"

        model, testdata = load_model_and_data()
        self.global_model = model
        self.max_rounds = rounds
        self.round = 0

        self.kafka_consumer = KafkaConsumer(
            bootstrap_servers=["0.0.0.0:9092"],
            consumer_timeout_ms=10000,  # Unblock thread
        )
        self.kafka_offset = 0
        self.pool = ThreadPoolExecutor(max_workers=4)

        self.test_data = testdata.batch(32)
        self.loss_buffer = list()
        self.federated_losses = list()
        self.global_losses = list()
        self.accuracy = list()

    def register_handles(self):
        self.sio.on("connect", self.connect)
        self.sio.on("fl_update", self.fl_update)

    async def connect(self, sid, environ):
        self.connected_nodes.append(sid)
        self.sio.enter_room(sid, self.training_room)

        async def start_training_callback():
            if len(self.connected_nodes) == self.min_nodes:
                print(f"Connected to {self.min_nodes}, starting training")
                await self.start_round()
            else:
                print(
                    f"Waiting to connect to "
                    f"{self.min_nodes - len(self.connected_nodes)} "
                    f"more nodes to start training"
                )

        await self.sio.emit(
            "connection_received",
            room=sid,
            callback=start_training_callback,
        )

    def run_server(self, host="0.0.0.0", port=5000):
        web.run_app(self.app, host=host, port=port)

    def store_history(self):  # Returns Keras model
        loss = reduce(lambda x,y:x+y, self.loss_buffer)
        self.federated_losses.append(loss)
        self.loss_buffer = list()

    def aggregate(self, client_mapped_weights):
        print("Aggregating updates")
        # sort weights

        all_weights = list()
        for client in client_mapped_weights:
            ordered_weights = list()
            ordered_layers = sorted(client_mapped_weights[client].keys())
            for layer_id in ordered_layers:
                ordered_weights.append(client_mapped_weights[client][layer_id])
            all_weights.append(ordered_weights)

        # aggregate weights
        agg_model_weights = {}
        for idx, layer in enumerate(self.global_model.layers):
            agg_layer_weights = []

            if layer.trainable_weights:
                for i in range(2):  # Aggregate weights and biases separately
                    lst_weights = [client[idx][i] for client in all_weights]
                    agg_layer_weights.append(
                        np.mean(np.array(lst_weights), axis=0)
                    )
            agg_model_weights.update({idx: agg_layer_weights})
        return agg_model_weights

    def evaluate(self, aggregated_weights):
        for idx, layer in enumerate(self.global_model.layers):
            if layer.trainable_weights:
                self.global_model.layers[idx].set_weights(aggregated_weights[idx])
        _, acc = self.global_model.evaluate(
            self.test_data, verbose=0
        )

    async def fl_update(self, sid, data):
        self.loss_buffer.append(data["metrics"]["loss"])
        self.pending_nodes.remove(sid)
        if not self.pending_nodes:
            loop = asyncio.get_event_loop()
            asyncio.ensure_future(self.async_consume(loop))

    def consume_updates(self):
        print("Consuming updates")
        count = 0

        # Create topic partition for our single partition setup
        tp = TopicPartition(topic="updates", partition=0)
        # Assign the topic to the consumer and seek the desired offset
        self.kafka_consumer.assign([tp])
        self.kafka_consumer.seek(tp, self.kafka_offset)

        client_mapped_weights = dict()
        
        for msg in self.kafka_consumer:
            count += 1
            client_name = msg.key.decode()
            client_msg = json.loads(msg.value)

            print(client_msg)
            if client_name not in client_mapped_weights:
                client_mapped_weights[client_name] = dict()
            client_mapped_weights[client_name].update(
                {client_msg["layer_id"]: decode(client_msg["model_layer"])}
            )

        aggregated_weights = self.aggregate(client_mapped_weights)
        self.evaluate(aggregated_weights)
        self.kafka_offset += count

    def async_consume(self, loop):
        yield from loop.run_in_executor(self.pool, self.consume_updates)
        # Hijack the event loop to enter end_round coroutine in the same thread
        # synchronously after all updates have been consumed.
        loop.create_task(self.end_round())

    async def start_round(self):
        print("Starting round")
        # Send model weights on Kafka
        self.pending_nodes = self.connected_nodes.copy()
        await self.sio.emit(
            "start_training",
            data={
                "model": self.global_model.to_json(),
                "weights": encode(self.global_model),
            },
            room=self.training_room,
        )

    async def end_round(self):
        print("Ending round")
        self.round += 1
        if self.round <= self.max_rounds:
            await self.start_round()
        else:
            await self.end_session()

    async def end_session(self):
        print("Ending session")
        await self.sio.emit(
            "end_session",
            room=self.training_room,
            data={
                "model": encode(self.global_model),
            },
        )

    async def disconnect(self, sid):
        self.connected_nodes.remove(sid)
        self.sio.leave_room(sid, room=self.training_room)


if __name__ == "__main__":
    fl_server = Server()
    fl_server.run_server()
