import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scikitplot
import sklearn
from sklearn.metrics import roc_auc_score, roc_curve

np.set_printoptions(linewidth=200)
directory = os.path.join(os.getcwd(), 'GNNout')


def file_list():

    # List of input GNN attempts, removing unwanted files from the list such as: previous image outputs and MacOS file

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

    # Plots the training history of the GNN with a plot of the loss and accuracy for both the training and validation
    # datasets

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
        ax.minorticks_on()
        ax.set(xlabel='Epochs', ylabel='Loss')

    plt.savefig(f'{directory}/{data_number}training_history.png', dpi=600)
    plt.clf()


def plot_sig_back_hist(predictions, targets, data_number):

    #  Plot a stacked histogram of how the signal and background has been classified by the GNN

    signal = np.where(targets == 1)[1]
    background = np.where(targets == 0)[1]
    signal_predictions = np.take(predictions, signal)  # List of probabilities that the GNN classed the event as signal
    background_predictions = np.take(predictions, background)  # List of prob for background

    plt.hist([signal_predictions, background_predictions], bins=100, stacked=True, label=['Signal', 'Background'])
    plt.legend()
    plt.xlabel('Prediction Value')
    plt.ylabel('N')
    plt.title('The distribution of prediction between \n 0 for background and 1 for signal')

    plt.savefig(f'{directory}/{data_number}sig_back_histogram.png', dpi=600)
    plt.clf()


def plot_sig_back_cumulative(predictions, targets, data_number):

    # Plot a cumulative plot of the classification of signal and background events compared to probability

    signal = np.where(targets == 1)[1]
    background = np.where(targets == 0)[1]
    signal_predictions = np.take(predictions, signal)  # List of probabilities that the GNN classed the event as signal
    background_predictions = np.take(predictions, background)  # List of prob for background

    signal_cumulative = np.cumsum(signal_predictions)
    background_cumulative = np.cumsum(background_predictions)

    fig, ax = plt.subplots(figsize=(5, 5))

    n, bins, patches = ax.hist(signal_predictions, bins=100, density=True, histtype='step', cumulative=True, label='Signal')
    patches[0].set_xy(patches[0].get_xy()[:-1])
    n, bins, patches = ax.hist(background_predictions, bins=bins, density=True, histtype='step', cumulative=-1, label='Background' )
    patches[0].set_xy(patches[0].get_xy()[1:])
    ax.minorticks_on()
    ax.legend()
    ax.set_xlabel('Probability events are classed as signal')
    ax.set_ylabel('Cumulative sum of events')

    plt.savefig(f'{directory}/{data_number}sig_back_cumulative.png', dpi=600)
    plt.clf()


def plot_sig_sqrt_back(predictions, targets, data_number):

    # Plot of how the s/sqrt(b) changes when the prediction cut changes

    thresholds = np.linspace(0, 1, 1001)
    sb_list = []

    for threshold in thresholds:
        combined = np.vstack((predictions, targets)).T
        combined = combined[combined[:, 0] > threshold]
        n_signal = np.count_nonzero(combined[:, 1] == 1)
        n_background = np.count_nonzero(combined[:, 1] == 0)
        sig = n_signal / np.sqrt(n_background)
        sb_list.append(sig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(thresholds, sb_list)
    ax.minorticks_on()
    ax.set_xlabel('Threshold')
    ax.set_ylabel(r'$s/\sqrt{b}$')

    plt.savefig(f'{directory}/{data_number}sig_sqrt_back.png', dpi=600)
    plt.clf()


def plot_roc_curve_plt(predictions, targets, data_number):

    # Plotting a Receiver Operating Characteristic (ROC) curve and working out the Area Underneath the Curve (AUC)

    thresholds = np.linspace(0, 1, 1001)

    tpr_list, fpr_list = [], []

    for threshold in thresholds:
        combined = np.vstack((predictions, targets)).T
        tttt = combined[combined[:, 0] > threshold]
        tt = combined[combined[:, 0] < threshold]
        tp = np.count_nonzero(tttt[:, 1] == 1)  # True positives
        fp = np.count_nonzero(tttt[:, 1] == 0)  # False positives
        tn = np.count_nonzero(tt[:, 1] == 0)  # True negative
        fn = np.count_nonzero(tt[:, 1] == 1)  # False negative

        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    plt.plot(fpr_list, tpr_list)
    plt.plot(fpr_list, fpr_list, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(f'{directory}/{data_number}roc_curve_plt.png', dpi=600)
    plt.clf()


def plot_roc_curve_skl(predictions, targets, data_number):

    # Plotting a ROC curve using the scikit learn Python library

    true_values = targets[0].astype(int)
    probabilities = predictions[0].astype(float)

    fpr, tpr, thresholds = roc_curve(true_values, probabilities)
    auc = roc_auc_score(true_values, probabilities)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect('equal')
    ax.plot(fpr, tpr, label='DNN AUC = %0.4f' % auc)
    ax.plot(fpr, fpr, linestyle='--', label='Random AUC = 0.5')
    ax.minorticks_on()
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(f'{directory}/{data_number}roc_curve_skl.png', dpi=600)
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
    plot_sig_back_cumulative(predictions, targets, data_info['t0'])
    plot_roc_curve_plt(predictions, targets, data_info['t0'])
    plot_sig_sqrt_back(predictions, targets, data_info['t0'])
    plot_roc_curve_skl(predictions, targets, data_info['t0'])
