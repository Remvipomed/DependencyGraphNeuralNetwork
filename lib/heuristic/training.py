
import clingo
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import tensorflow.keras.backend as k
import spektral

from tensorflow.keras.losses import Hinge
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import Adam

from . import gnn_model
from ..encoding import graph


class GraphNeuralNetworkHeuristic:

    def __init__(self):
        self.history = None

    def train_dataset(self, dataset):
        loader = spektral.data.SingleLoader(dataset)
        model = spektral.models.gcn.GCN(
            1,
            channels=16,
            activation='relu',
            output_activation='tanh',
            use_bias=True,
            dropout_rate=0.5,
            l2_reg=0.00025
        )

        learning_rate = 1e-3
        optimizer = Adam(learning_rate)
        loss_fn = Hinge()
        metric = accuracy_fn

        model.compile(optimizer=optimizer, loss=loss_fn, metrics=[metric])

        self.history = model.fit(loader.load(), steps_per_epoch=loader.steps_per_epoch, epochs=1000)
        # self.show_history()

    def show_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(self.history.history["loss"])
        # ax1.plot(self.history.history["val_loss"])
        ax1.legend(["train"], loc="upper right")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")

        ax2.plot(self.history.history["accuracy"])
        # ax2.plot(self.history.history["val_acc"])
        ax2.legend(["train"], loc="upper right")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Accuracy")
        plt.show()


def accuracy_fn(y_true, y_pred):
    y_pred = k.sign(y_pred)
    y_true = k.sign(y_true)
    return k.mean(k.equal(y_true, y_pred))


def train_graph_neural_network(dataset):
    heuristic = GraphNeuralNetworkHeuristic()
    print("start training")
    heuristic.train_dataset(dataset)
    print("end training")
