import os
import numpy as np
import tensorflow as tf
import time
import pickle

import spektralDataset
from Layers import MessagePassing

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import categorical_accuracy, binary_accuracy, BinaryAccuracy
from tensorflow.keras.optimizers import Adam

from spektral.data import DisjointLoader
from spektral.layers import GlobalAvgPool, ECCConv

np.set_printoptions(linewidth=200)

################################################################################
# PARAMETERS
################################################################################

learning_rate = 1e-3  # Learning rate
epochs = 2000  # Number of training epochs
batch_size = 32  # Batch size
es_patience = 20  # Patience for early stopping
samples = 24000  # Number of graphs to add to the dataset
channels = 256  # Number of channels in each layer of the ECCConv
t0 = time.time()

################################################################################
# LOAD DATA
################################################################################

file = ['root2networkOut/0 graphs.npz', 'root2networkOut/1 graphs.npz']
dataset = spektralDataset.TopDataset(max_samples=samples, file_name_list=file)

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

X_1 = ECCConv(channels, activation='relu')([X_in, A_in, E_in])
X_2 = ECCConv(channels, activation='relu')([X_1, A_in, E_in])
X_3 = ECCConv(channels, activation='relu')([X_2, A_in, E_in])
X_4 = GlobalAvgPool()([X_3, I_in])
output = Dense(n_out, activation='sigmoid')(X_4)

# Build model
model = Model(inputs=[X_in, A_in, E_in, I_in], outputs=output)
opt = Adam(lr=learning_rate)
loss_fn = BinaryCrossentropy()

# Saving the model summary for later use
model.summary()
current_directory = os.getcwd()
final_directory = os.path.join(current_directory, 'GNNout')  # Save keras model details as a txt file
filename_model = os.path.join(final_directory, f'{t0}GNNmodel.txt')
with open(filename_model, 'wt') as text_file_model:
    model.summary(print_fn=lambda line: text_file_model.write(line + '\n'))
text_file_model.close()

filename_details = os.path.join(final_directory, f'{t0}GNNdetails.txt')  # Save hyperparameters as a txt file
text_file_details = open(filename_details, 'wt')
text_file_details.write(f'Learning Rate: {learning_rate} | Epochs: {epochs} | Batch Size: {batch_size} | '
                        f'Early Stopping Patience: {es_patience}\nNumber of Samples: {samples} | Channels: {channels}')
text_file_details.close()

################################################################################
# FIT MODEL
################################################################################

loss_values, val_loss_values, accuracy_values, val_accuracy_values = [], [], [], []


@tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
def train_step(inputs, target):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(target, predictions)
        loss += sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    acc = tf.reduce_mean(binary_accuracy(tf.one_hot(target, 1), predictions))
    return loss, acc


def test_step(loader):
    prediction_list, target_list = [], []
    step = 0
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()
        predictions = model(inputs, training=False)
        targets = tf.cast(target, tf.float32)

        prediction_list.append(predictions)
        target_list.append(targets)

    y_reco = tf.concat(prediction_list, axis=0).numpy()
    y_true = tf.concat(target_list, axis=0).numpy()
    return y_reco, y_true


def evaluate(loader):
    output_l = []
    step = 0
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()
        pred = model(inputs, training=False)
        outs = (loss_fn(target, pred), tf.reduce_mean(binary_accuracy(tf.one_hot(target, 1), pred)),)
        output_l.append(outs)
    return np.mean(output_l, 0)


print("Fitting model")
current_batch = epoch = model_loss = model_acc = best_val_epoch = 0
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

        val_loss_values.append(val_loss)
        val_accuracy_values.append(val_acc)
        loss_values.append(model_loss)
        accuracy_values.append(model_acc)

        # Check if loss improved for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch
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
test_pre, test_target = test_step(loader_te)
print(test_pre)
print(test_target)

predictions = test_pre.T
targets = test_target.T
signal = np.where(targets == 1)[1]
background = np.where(targets == 0)[1]
signal_predictions = np.take(predictions, signal)
background_predictions = np.take(predictions, background)

# Saving for later processing

filename_predictions = os.path.join(final_directory, f'{t0}Predictions')
np.savez(filename_predictions, predictions=predictions, targets=targets)  # Predictions and targets

filename_dictionary = os.path.join(final_directory, f'{t0}GNNdictionary.pkl')  # Details about the model
dictionary = {"Learning Rate": learning_rate, "Epochs": epochs, "Batch Size": batch_size,
              "Early Stopping Patience": es_patience, 'Samples': samples, "Best Value Epoch": best_val_epoch,
              "Test Loss": test_loss, "Test Accuracy": test_acc, "t0": t0, "Channels": channels}
with open(filename_dictionary, 'wb') as text_file_dict:
    pickle.dump(dictionary, text_file_dict, protocol=0)

filename_training = os.path.join(final_directory, f'{t0}Training')
np.savez(filename_training, loss_values=loss_values, val_loss_values=val_loss_values,
         accuracy_values=accuracy_values, val_accuracy_values=val_accuracy_values)

print("Done. Test loss: {:.4f}. Test acc: {:.2f}".format(test_loss, test_acc))

print(f'Finished GNN: \nTime taken: {round(time.time() - t0, 5)}s')
