
import json
import clingo
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import keras.backend as k
import spektral

from spektral.models import GCN
from keras.models import model_from_json

from .. import project_root
from . import gnn_model
from ..encoding import graph
from ..encoding.dataset import DemoSet


model_path = project_root.joinpath("data/models")


class GraphNeuralNetworkHeuristic:

    def __init__(self, name):
        self.name = name
        self.input_shape = None
        self.model = gnn_model.create_model()
        self.history = None

    def save(self):
        path_config = model_path.joinpath(f"{self.name}_config.json")
        path = model_path.joinpath(f"{self.name}.h5")
        config = {**self.model.get_config(), "input_shape": self.input_shape}
        path_config.write_text(json.dumps(config))
        self.model.save_weights(path)

    def load(self):
        path_config = model_path.joinpath(f"{self.name}_config.json")
        path = model_path.joinpath(f"{self.name}.h5")

        model_config = json.loads(path_config.read_text())
        input_shape = model_config.pop("input_shape")
        self.model.from_config(model_config)
        x_input = tf.zeros(dtype=tf.float32, shape=input_shape)
        a_input = tf.zeros(dtype=tf.float32, shape=(input_shape[0], input_shape[0]))
        self.model([x_input, a_input])
        self.model.load_weights(path)

    def predict(self, ctl: clingo.Control):
        dependency_graph = graph.create_encoded_graph(ctl)
        spek_graph = graph.create_spektral_graph(dependency_graph)
        prediction = self.model.predict([spek_graph])
        return prediction

    def evaluate(self, ctl: clingo.Control):
        dependency_graph = graph.create_labeled_graph(ctl)
        spek_graph = graph.create_spektral_graph(dependency_graph)
        labels = spek_graph.y

        input_data = [spek_graph.x, spek_graph.a]
        prediction = self.model.predict(input_data)
        prediction_class = np.sign(np.squeeze(prediction))

        labels_positive = (labels == 1)
        labels_negative = (labels == -1)
        prediction_positive = (prediction_class == 1) & labels_positive
        prediction_negative = (prediction_class == -1) & labels_negative
        print("positive accuracy:", np.count_nonzero(prediction_positive), "/", np.count_nonzero(labels_positive))
        print("negative accuracy:", np.count_nonzero(prediction_negative), "/", np.count_nonzero(labels_negative))

    def train_dataset(self, dataset):
        self.input_shape = dataset[0].x.shape
        labels = dataset.graphs[0]["y"]
        print(np.count_nonzero(labels == 1), "/", len(labels))
        loader = spektral.data.SingleLoader(dataset)

        epochs = 100
        self.history = self.model.fit(loader.load(), steps_per_epoch=loader.steps_per_epoch, epochs=epochs)
        self.show_history()

    def show_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(self.history.history["loss"])
        # ax1.plot(self.history.history["val_loss"])
        ax1.legend(["train"], loc="upper right")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")

        ax2.plot(self.history.history["accuracy_fn"])
        # ax2.plot(self.history.history["val_acc"])
        ax2.legend(["train"], loc="upper right")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Accuracy")
        plt.show()


def train_graph_neural_network():
    dataset = DemoSet()
    heuristic = GraphNeuralNetworkHeuristic("demo")
    print("start training")
    heuristic.train_dataset(dataset)
    heuristic.save()
    print("end training")
