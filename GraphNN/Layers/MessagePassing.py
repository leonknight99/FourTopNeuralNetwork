from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.layers.core import Dropout

import inspect

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

from spektral.layers.ops.scatter import deserialize_scatter, serialize_scatter
from spektral.utils.keras import (
    deserialize_kwarg,
    is_keras_kwarg,
    is_layer_kwarg,
    serialize_kwarg,
)

# class TopQuarkMP(MessagePassing):
#
#     def __init__(self, layers):
#         super().__init__()
#         self.layers = layers
#         self.hidden_states = self.n_nodes
#         self.message_mlp = MLP(self.hidden_states, layers=self.layers)
#         self.update_mlp = MLP(self.hidden_states, layers=1)
#
#     def message(self, x, **kwargs):
#         # print([self.get_i(x), self.get_j(x), e])
#         out = self.message_mlp(x)
#         return out
#
#     def aggregate(self, messages, **kwargs):
#         return
#
#     def update(self, embeddings, training=False):
#         out = self.update_mlp(embeddings, training=training)
#         return out


class TopQuarkMP(Layer):

    def __init__(self, aggregate="sum", **kwargs):
        super().__init__(**{k: v for k, v in kwargs.items() if is_keras_kwarg(k)})
        self.kwargs_keys = []
        for key in kwargs:
            if is_layer_kwarg(key):
                attr = kwargs[key]
                attr = deserialize_kwarg(key, attr)
                self.kwargs_keys.append(key)
                setattr(self, key, attr)

        self.msg_signature = inspect.signature(self.message).parameters
        self.agg_signature = inspect.signature(self.aggregate).parameters
        self.upd_signature = inspect.signature(self.update).parameters
        self.agg = deserialize_scatter(aggregate)

        print(self.msg_signature)
        print(self.agg_signature)
        print(self.upd_signature)

        #self.message_mlp = MLP(self.hidden_states, layers=1)
        #self.update_mlp = MLP(self.hidden_states, layers=1)

    def call(self, inputs, **kwargs):
        x, a, e = self.get_inputs(inputs)
        return self.propagate(x, a, e)

    def build(self, input_shape):
        self.built = True

    def propagate(self, x, a, e=None, **kwargs):
        self.n_nodes = tf.shape(x)[-2]
        self.index_i = a.indices[:, 1]
        self.index_j = a.indices[:, 0]

        # Message
        msg_kwargs = self.get_kwargs(x, a, e, self.msg_signature, kwargs)
        messages = self.message(x, **msg_kwargs)

        # Aggregate
        agg_kwargs = self.get_kwargs(x, a, e, self.agg_signature, kwargs)
        embeddings = self.aggregate(messages, **agg_kwargs)

        # Update
        upd_kwargs = self.get_kwargs(x, a, e, self.upd_signature, kwargs)
        output = self.update(embeddings, **upd_kwargs)

        return output

    def message(self, x, **kwargs):
        return self.get_j(x)

    def aggregate(self, messages, **kwargs):
        return self.agg(messages, self.index_i, self.n_nodes)

    def update(self, embeddings, **kwargs):
        return embeddings

    def get_i(self, x):
        return tf.gather(x, self.index_i, axis=-2)

    def get_j(self, x):
        return tf.gather(x, self.index_j, axis=-2)

    def get_kwargs(self, x, a, e, signature, kwargs):
        output = {}
        for k in signature.keys():
            if signature[k].default is inspect.Parameter.empty or k == "kwargs":
                pass
            elif k == "x":
                output[k] = x
            elif k == "a":
                output[k] = a
            elif k == "e":
                output[k] = e
            elif k in kwargs:
                output[k] = kwargs[k]
            else:
                raise ValueError("Missing key {} for signature {}".format(k, signature))

        return output

    @staticmethod
    def get_inputs(inputs):
        """
        Parses the inputs lists and returns a tuple (x, a, e) with node features,
        adjacency matrix and edge features. In the inputs only contain x and a, then
        e=None is returned.
        """
        if len(inputs) == 3:
            x, a, e = inputs
            assert K.ndim(e) in (2, 3), "E must have rank 2 or 3"
        elif len(inputs) == 2:
            x, a = inputs
            e = None
        else:
            raise ValueError(
                "Expected 2 or 3 inputs tensors (X, A, E), got {}.".format(len(inputs))
            )
        assert K.ndim(x) in (2, 3), "X must have rank 2 or 3"
        assert K.is_sparse(a), "A must be a SparseTensor"
        assert K.ndim(a) == 2, "A must have rank 2"

        return x, a, e

    def get_config(self):
        mp_config = {"aggregate": serialize_scatter(self.agg)}
        keras_config = {}
        for key in self.kwargs_keys:
            keras_config[key] = serialize_kwarg(key, getattr(self, key))
        base_config = super().get_config()

        return {**base_config, **keras_config, **mp_config, **self.config}

    @property
    def config(self):
        return {}

    @staticmethod
    def preprocess(a):
        return a

class MLP(Model):
    def __init__(self, hidden, layers=1, batch_norm=True, dropout=0.0, activation='relu'):
        super().__init__()
        self.batch_norm = batch_norm
        self.dropout_rate = dropout

        self.mlp = Sequential()
        for i in range(layers):
            # Linear
            self.mlp.add(Dense(hidden, activation=activation))
            if dropout > 0:
                self.mlp.add(Dropout(dropout))

    def call(self, inputs, training=False):
        return self.mlp(inputs, training=training)
