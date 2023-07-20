
import spektral
import tensorflow as tf
import keras.backend as k

from spektral.models import GCN
from spektral.layers import GCNConv, GlobalSumPool

from keras.losses import Hinge
from keras.metrics import CategoricalAccuracy
from keras.optimizers import Adam
from keras.models import load_model


HIDDEN_LAYERS = 32
CLASSES = 1


def create_model():
    model = spektral.models.gcn.GCN(
        n_labels=1,
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
    return model


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


def accuracy_fn(y_true, y_pred):
    y_pred = k.sign(y_pred)
    y_true = k.sign(y_true)
    return k.mean(k.equal(y_true, y_pred))
