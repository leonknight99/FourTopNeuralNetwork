import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import spektralDataset

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import categorical_accuracy
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
es_patience = 10  # Patience for early stopping
t0 = time.time()

################################################################################
# LOAD DATA
################################################################################

file = ['root2networkOut/0 graphs.npz', 'root2networkOut/1 graphs.npz']
dataset = spektralDataset.TopDataset(max_samples=16384, file_name_list=file)

print(f'Time taken: {round(time.time() - t0, 5)}s | Data opened')

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
loader_tr = DisjointLoader(dataset_tr, batch_size=batch_size, epochs=epochs)
loader_va = DisjointLoader(dataset_va, batch_size=batch_size)
loader_te = DisjointLoader(dataset_te, batch_size=batch_size)

print(f'Time taken: {round(time.time() - t0, 5)}s | Loaders complete')

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
        predictions = model(inputs, training=True)
        loss = loss_fn(target, predictions)
        loss += sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    acc = tf.reduce_mean(categorical_accuracy(target, predictions))
    return loss, acc


def evaluate(loader):
    output_l = []
    step = 0
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()
        pred = model(inputs, training=False)
        outs = (loss_fn(target, pred),tf.reduce_mean(categorical_accuracy(target, pred)),)
        output_l.append(outs)
    return np.mean(output_l, 0)


print("Fitting model")
current_batch = epoch = model_loss = model_acc = 0
best_val_loss = np.inf
best_weights = None
patience = es_patience

for batch in loader_tr:
    outs = train_step(*batch)

    model_loss += outs[0]
    model_acc += outs[1]
    current_batch += 1
    if current_batch == loader_tr.steps_per_epoch:
        model_loss /= loader_tr.steps_per_epoch
        model_acc /= loader_tr.steps_per_epoch
        epoch += 1

        # Compute validation loss and accuracy
        val_loss, val_acc = evaluate(loader_va)
        print("Ep. {} - Loss: {:.2f} - Acc: {:.2f} - Val loss: {:.2f} - Val acc: {:.2f}"
              .format(epoch, model_loss, model_acc, val_loss, val_acc))

        # Check if loss improved for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = es_patience
            print("New best val_loss {:.3f}".format(val_loss))
            best_weights = model.get_weights()
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping (best val_loss: {})".format(best_val_loss))
                break
        model_loss = 0
        model_acc = 0
        current_batch = 0

################################################################################
# EVALUATE MODEL
################################################################################
print("Testing model")
model.set_weights(best_weights)  # Load best model
test_loss, test_acc = evaluate(loader_te)
print("Done. Test loss: {:.4f}. Test acc: {:.2f}".format(test_loss, test_acc))

history = model.history.history

loss_values = history['loss']
val_loss_values = history['val_loss']
accuracy_values = history['accuracy']
val_accuracy_values = history['val_accuracy']

fig, axs = plt.subplots(1, 2, figsize=(10,10))

epochs = range(1, epochs + 1)

axs[0].plot(epochs, loss_values, 'bo', label='Training loss')
axs[0].plot(epochs, val_loss_values, 'b', label='Validation loss')
axs[0].title.set_text(f'Training and validation loss for {epoch_n_val} epochs')
#axs[0].xlabel.set_text('Epochs')
#axs[0].ylabel.set_text('Loss')
axs[0].legend()

axs[1].plot(epochs, accuracy_values, 'bo', label='Training acc')
axs[1].plot(epochs, val_accuracy_values, 'b', label='Validation acc')
axs[1].title.set_text(f'Training and validation accuracy for {epoch_n_val} epochs')
#axs[1].xlabel.set_text('Epochs')
#axs[1].ylabel.set_text('Loss')
axs[1].legend()


