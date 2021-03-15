import numpy as np
import tensorflow as tf
import spektralDataset

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from spektral.data import DisjointLoader, BatchLoader
from spektral.transforms import NormalizeAdj
from spektral.datasets import QM9
from spektral.layers import ECCConv, GlobalSumPool

np.set_printoptions(linewidth=200)

################################################################################
# PARAMETERS
################################################################################
learning_rate = 1e-3  # Learning rate
epochs = 100  # Number of training epochs
batch_size = 32  # Batch size

################################################################################
# LOAD DATA
################################################################################

file = ['root2networkOut/0 graphs.npz', 'root2networkOut/1 graphs.npz']
dataset = spektralDataset.TopDataset(max_samples=16384, file_name_list=file)
lista, liste = [], []
'''for n in range(512):
    lista.append(dataset[n].a.nnz)
    liste.append(dataset[n].e.shape[0])

print(dataset[214])
print(dataset[214].a.nnz)
print(dataset[214].x.shape)
print(dataset[214].e.shape)
print(dataset[214].a)
print(dataset[214].x)
print(dataset[214].e)
print(dataset[214].y)

print(lista)
print(liste)
print(lista == liste)
breakpoint()'''
# Parameters
F = dataset.n_node_features  # Dimension of node features
S = dataset.n_edge_features  # Dimension of edge features
n_out = dataset.n_labels  # Dimension of the target

print(F, S, n_out)

# Train/test split
idxs = np.random.permutation(len(dataset))
split = int(0.9 * len(dataset))
idx_tr, idx_te = np.split(idxs, [split])
dataset_tr, dataset_te = dataset[idx_tr], dataset[idx_te]

loader_tr = DisjointLoader(dataset_tr, batch_size=batch_size, epochs=epochs)
loader_te = DisjointLoader(dataset_te, batch_size=batch_size, epochs=1)
print(loader_tr.tf_signature())
print(loader_te.tf_signature())

################################################################################
# BUILD MODEL
################################################################################
X_in = Input(shape=(None,F), name="X_in")
A_in = Input(shape=(None,), sparse=True, name="A_in")
E_in = Input(shape=(None,S), name="E_in")
I_in = Input(shape=(), name="segment_ids_in", dtype=tf.int32)

X_1 = ECCConv(32, activation="relu")([X_in, A_in, E_in])
X_2 = ECCConv(32, activation="relu")([X_1, A_in, E_in])
X_3 = GlobalSumPool()([X_2, I_in])
output = Dense(n_out)(X_3)

# Build model
model = Model(inputs=[X_in, A_in, E_in, I_in], outputs=output)
opt = Adam(lr=learning_rate)
loss_fn = MeanSquaredError()

model.summary()

################################################################################
# FIT MODEL
################################################################################
#@tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
def train_step(inputs, target):
    with tf.GradientTape() as tape:
        x, e, a, i = inputs
        #print(f'{x}\n{x.shape}\n{tf.shape(x)}\n\n\n{e}\n{e.shape}\n{tf.shape(e)}\n\n\n{a}\n{a.shape}\n{tf.shape(a)}\n\n\n{i}\n{i.shape}\n{tf.shape(i)}')
        #print(f'\n\n{target}\n{len(target)}')
        predictions = model(inputs, training=True)
        #print(f'Predictions:\n{predictions}')
        loss = loss_fn(target, predictions)
        loss += sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


print("Fitting model")
current_batch = 0
model_loss = 0

for batch in loader_tr:
    inputs, target = batch
    x, e, a, i = inputs
    #print(f'{x}\n{x.shape}\n{tf.shape(x)}\n\n\n{e}\n{e.shape}\n{tf.shape(e)}\n\n\n{a}\n{a.shape}\n{tf.shape(a)}\n\n\n{i}\n{i.shape}\n{tf.shape(i)}')
    #print(f'\n\n{target}\n{len(target)}')
    #tf.debugging.enable_check_numerics()
    #tf.print(*batch)
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
print("Testing model")
model_loss = 0
for batch in loader_te:
    inputs, target = batch
    predictions = model(inputs, training=False)
    model_loss += loss_fn(target, predictions)
model_loss /= loader_te.steps_per_epoch
print("Done. Test loss: {}".format(model_loss))