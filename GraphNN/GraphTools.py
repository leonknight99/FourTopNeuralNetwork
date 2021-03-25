import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=200)

directory = os.path.join(os.getcwd(), 'GNNout')


def file_list():
    input_files = sorted(os.listdir(directory))
    input_files = [x for x in input_files if '.png' not in x]
    try:
        input_files.remove('.DS_Store')
    except ValueError:
        pass
    input_files = np.array(input_files).reshape((-1, 4))
    print(input_files)
    return input_files


def plot_history(loss_values, val_loss_values, accuracy_values, val_accuracy_values, best_val_epoch, data_number):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    epoch_list = range(1, len(loss_values) + 1)

    axs[0].plot(epoch_list, loss_values, 'c', label='Training loss')
    axs[0].plot(epoch_list, val_loss_values, 'b', label='Validation loss')
    axs[0].axvline(x=best_val_epoch, label='Best validation loss epoch', c='r')
    axs[0].title.set_text(f'Training and validation loss')
    axs[0].legend()

    axs[1].plot(epoch_list, accuracy_values, 'c', label='Training accuracy')
    axs[1].plot(epoch_list, val_accuracy_values, 'b', label='Validation accuracy')
    axs[1].axvline(x=best_val_epoch, label='Best validation loss epoch', c='r')
    axs[1].title.set_text(f'Training and validation accuracy')
    axs[1].legend()

    for ax in axs.flat:
        ax.set(xlabel='Epochs', ylabel='Loss')

    plt.savefig(f'{directory}/{data_number}training_history.png', dpi=600)
    plt.clf()


def plot_sig_back_hist(predictions, targets, data_number):
    signal = np.where(targets == 1)[1]
    background = np.where(targets == 0)[1]
    signal_predictions = np.take(predictions, signal)
    background_predictions = np.take(predictions, background)

    plt.hist([signal_predictions, background_predictions], bins=100, stacked=True, label=['signal', 'background'])
    plt.legend()
    plt.xlabel('Prediction Value')
    plt.ylabel('N')
    plt.title('The distribution of prediction between \n 0 for background and 1 for signal')

    plt.savefig(f'{directory}/{data_number}sig_back_histogram.png', dpi=600)
    plt.clf()

files = file_list()
for j in range(files.shape[0]):
    pickle_file = open(f'{directory}/{files[j,0]}', 'rb')
    data_info = pickle.load(pickle_file)

    data_predictions = np.load(f'{directory}/{files[j,2]}')  # Predictions data
    predictions, targets = data_predictions['predictions'], data_predictions['targets']

    data_training = np.load(f'{directory}/{files[j,3]}')  # History of learning data
    loss_values, val_loss_values, accuracy_values, val_accuracy_values = data_training['loss_values'], \
                                                                         data_training['val_loss_values'], \
                                                                         data_training['accuracy_values'], \
                                                                         data_training['val_accuracy_values']

    plot_history(loss_values, val_loss_values, accuracy_values, val_accuracy_values,
                 int(data_info['Best Value Epoch']), data_info['t0'])
    plot_sig_back_hist(predictions, targets, data_info['t0'])

