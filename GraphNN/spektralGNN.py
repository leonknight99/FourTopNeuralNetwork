import numpy as np
import time
import tensorflow as tf
import spektralDataset
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from spektral.data import DisjointLoader, BatchLoader
from spektral.layers import MessagePassing
from spektral.transforms.normalize_adj import NormalizeAdj
from spektral.transforms.adj_to_sp_tensor import AdjToSpTensor

np.set_printoptions(linewidth=200)

'''
    Parameters
'''
learning_rate = 1e-3
epochs = 100
batch_size = 32
t0 = time.time()

'''
    Load Data
'''
file = ['root2networkOut/0 graphs.npz', 'root2networkOut/1 graphs.npz']
dataset = spektralDataset.TopDataset(max_samples=0, file_name_list=file)

print(dataset[0])
print(dataset[0].a)
print(type(dataset[0].a))
print(dataset[0].x)
print(dataset[0].e)
print(dataset[0].y)

print(f'Time taken: {round(time.time() - t0, 5)}s | Data loaded')

# Parameters:
F = dataset.n_node_features  # Dimension of node features
S = dataset.n_edge_features  # Dimension of edge features
n_out = dataset.n_labels  # Dimension of the target

idxs = np.random.permutation(len(dataset))
split_va, split_te = int(0.8 * len(dataset)), int(0.9 * len(dataset))
idx_tr, idx_va, idx_te = np.split(idxs, [split_va, split_te])
dataset_tr, dataset_va, dataset_te = dataset[idx_tr], dataset[idx_va], dataset[idx_te]

print(f'Time taken: {round(time.time() - t0, 5)}s | Data split')

# Loaders:
loader_tr = DisjointLoader(dataset_tr, batch_size=batch_size, epochs=epochs, node_level=False)
loader_va = DisjointLoader(dataset_va, batch_size=batch_size, epochs=1, node_level=False)
loader_te = DisjointLoader(dataset_te, batch_size=batch_size, epochs=1, node_level=False)

print(f'Time taken: {round(time.time() - t0, 5)}s | Loaders complete')


X_in = Input(shape=(F, ), name='X_in')
A_in = Input(shape=(None, ), sparse=True, name='A_in')
E_in = Input(shape=(S, ), name='E_in')
I_in = Input(shape=(), name='segment_ids_in', dtype=tf.int64)

X_1 = MessagePassing(aggregate="mean")([X_in, A_in, E_in])
X_2 = MessagePassing(aggregate="mean")([X_1, A_in, E_in])
X_3 = MessagePassing(aggregate="mean")([X_2, A_in, E_in])
output = Dense(n_out)(X_3)

# Build model
model = Model(inputs=[X_in, A_in, E_in, I_in], outputs=output)  # I_in
opt = Adam(lr=learning_rate)
loss_fn = BinaryCrossentropy()
acc_fn = BinaryCrossentropy()


#@tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
def train_step(inputs, target):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(target, predictions)
        loss += sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


print(f'Time taken: {round(time.time() - t0, 5)}s | Fitting model')
current_batch = 0
model_loss = 0
for batch in loader_tr:
    outs = train_step(*batch)

    model_loss += outs
    current_batch += 1
    if current_batch == loader_tr.steps_per_epoch:
        print("Loss: {}".format(model_loss / loader_tr.steps_per_epoch))
        model_loss = 0
        current_batch = 0

################################################################################
# EVALUATE MODEL
################################################################################
print(f'Time taken: {round(time.time() - t0, 5)}s | Testing model')
model_loss = 0
for batch in loader_te:
    inputs, target = batch
    predictions = model(inputs, training=False)
    model_loss += loss_fn(target, predictions)
model_loss /= loader_te.steps_per_epoch
print("Done. Test loss: {}".format(model_loss))
