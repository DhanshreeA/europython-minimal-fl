This code is part of a [talk](https://youtu.be/fkLgRhcyPSU) delivered at EuroPython 2021. 


**This demo is based on the Keras [tutorial](https://keras.io/examples/structured_data/structured_data_classification_from_scratch/) on structured data**

This code is tested with Python 3.7.11, other Python versions may not work properly.

To run this demo, make sure you have docker and docker-compose set up.

Find the installation instructions for the above tools here: https://docs.docker.com/compose/install/ and https://docs.docker.com/compose/install/

Run the kafka service through docker-compose as `docker-compose up -d`

Now open a new terminal and run `python server.py` (You can configure the number of rounds and nodes)

Now open a new termnial corresponding to the number of nodes you have configured in the server and run `python client.py` in each terminal. 

