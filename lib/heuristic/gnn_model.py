
import tensorflow as tf

from spektral.models import GCN
from spektral.layers import GCNConv, GlobalSumPool


HIDDEN_LAYERS = 32
CLASSES = 2


class AtomHeuristicGNN(GCN):

    def __init__(self, n_features):
        x_in = tf.keras.layers.Input(shape=(n_features,))
        a_in = tf.keras.layers.Input(shape=(None,), sparse=True)

        super().__init__(x_in, a_in)

        self.conv1 = GCNConv(HIDDEN_LAYERS, activation='relu')
        self.conv2 = GCNConv(HIDDEN_LAYERS, activation='relu')
        self.dense = tf.keras.layers.Dense(CLASSES, activation='softmax')

    def call(self, inputs):
        x, a, i = inputs
        x = self.conv1([x, a])
        x = self.conv2([x, a])
        x = self.dense(x)
        return x
