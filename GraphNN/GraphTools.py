import os
import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

np.set_printoptions(linewidth=200)
directory = os.path.join(os.getcwd(), 'GNNout')


def smooth_function(y, width):
    if width % 2 == 0:
        print('Enter odd value for width of smoothing')
        return
    smooth = np.zeros_like(y)
    pm = math.floor(width / 2)
    for val in range(pm, len(y) - 1):
        smooth[val] = np.average(y[val-pm:val+pm+1])
    for val in range(0, pm):
        smooth[val] = y[val]
        smooth[len(y) - val - 1] = y[len(y) - val - 1]
    return smooth


def file_list():

    # List of input GNN attempts, removing unwanted files from the list such as: previous image outputs and MacOS file

    input_files = sorted(os.listdir(directory))
    input_files = [x for x in input_files if '.png' not in x]  # Removes previous outputted images
    input_files = [x for x in input_files if '.txt' not in x]  # Removes txt files used for GNN properties
    try:
        input_files.remove('.DS_Store')
    except ValueError:
        pass
    input_files = np.array(input_files).reshape((-1, 3))
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
    axs[0].legend(prop={'size': 6})

    axs[1].plot(epoch_list, accuracy_values, 'c', label='Training accuracy')
    axs[1].plot(epoch_list, val_accuracy_values, 'b', label='Validation accuracy')
    axs[1].axvline(x=best_val_epoch, label='Best validation loss epoch', c='r')
    axs[1].title.set_text(f'Training and validation accuracy')
    axs[1].legend(prop={'size': 6})

    for ax in axs.flat:
        ax.minorticks_on()
        ax.set(xlabel='Epochs', ylabel='Loss')

    plt.savefig(f'{directory}/{data_number}training_history.png', dpi=600)
    plt.clf()
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 5))

    smooth = smooth_function(np.log(val_loss_values), 5)

    ax.plot(epoch_list, np.log(loss_values), 'c', label='Training data loss')
    ax.plot(epoch_list, np.log(val_loss_values), 'b', label='Validation data loss')
    ax.plot(epoch_list, smooth, '--', label='Validation data "smoothed" over 5 epochs')
    ax.axvline(x=best_val_epoch, label='Best epoch', c='r', linewidth=0.5)
    ax.minorticks_on()
    ax.set_xlabel('Epochs')
    ax.set_ylabel(r'$\ln({loss})$')
    plt.legend(prop={'size': 6})

    plt.savefig(f'{directory}/{data_number}loss_history.png', dpi=600)
    plt.clf()
    plt.close()


def plot_sig_back_hist(predictions, targets, data_number):

    #  Plot a stacked histogram of how the signal and background has been classified by the GNN

    signal = np.where(targets == 1)[1]
    background = np.where(targets == 0)[1]
    signal_predictions = np.take(predictions, signal)  # List of probabilities that the GNN classed the event as signal
    background_predictions = np.take(predictions, background)  # List of prob for background

    plt.hist([signal_predictions, background_predictions], bins=100, stacked=True, label=['Signal', 'Background'])
    plt.legend(prop={'size': 6})
    plt.xlabel('Prediction Value')
    plt.ylabel('N')
    plt.title('The distribution of prediction between \n 0 for background and 1 for signal')

    plt.savefig(f'{directory}/{data_number}sig_back_histogram.png', dpi=600)
    plt.clf()
    plt.close()


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
    ax.legend(prop={'size': 6})
    ax.set_xlabel('Probability events are classed as signal')
    ax.set_ylabel('Cumulative sum of events')

    plt.savefig(f'{directory}/{data_number}sig_back_cumulative.png', dpi=600)
    plt.clf()
    plt.close()


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

    max_significance = max(sb_list)
    max_threshold = thresholds[sb_list.index(max_significance)]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(thresholds, sb_list, label='Significance')
    ax.plot(max_threshold, max_significance, '.', color='r', label=f'Maximum ({max_threshold:0.2f} , {max_significance:0.2f})')
    ax.minorticks_on()
    ax.set_xlabel('Threshold')
    ax.set_ylabel(r'$s/\sqrt{b}$')

    plt.legend(prop={'size': 6})
    plt.savefig(f'{directory}/{data_number}sig_sqrt_back.png', dpi=600)
    plt.clf()
    plt.close()


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
    plt.close()


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
    plt.legend(prop={'size': 6})
    plt.savefig(f'{directory}/{data_number}roc_curve_skl.png', dpi=600)
    plt.clf()
    plt.close()


def plot_roc_auc_epoch_variation(predict_list, targ_list, epoch_list, best_epoch_list):

    # SMPL combining roc curves for a variety of epoch training values

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect('equal')

    x = np.linspace(0, 1, 100)

    for n in range(len(predict_list)):
        predictions = predict_list[n]
        targets = targ_list[n]
        epoch = epoch_list[n]
        best_epoch = best_epoch_list[n]

        if epoch == 2000:
            epoch = best_epoch

        true_values = targets[0].astype(int)
        probabilities = predictions[0].astype(float)

        fpr, tpr, thresholds = roc_curve(true_values, probabilities)
        auc = roc_auc_score(true_values, probabilities)

        ax.plot(fpr, tpr, label=f'{epoch} AUC = {auc:0.4f}')

    ax.plot(x, x, linestyle='--', label='Random AUC = 0.5')
    ax.minorticks_on()
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    plt.legend(prop={'size': 6})
    plt.savefig(f'{directory}/roc_auc_epoch_variation.png', dpi=600)
    plt.clf()
    plt.close()


def plot_roc_auc_channels(predict_list, targ_list, channel_list):

    # ECCConv combining roc curves for a variety of channel values

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect('equal')

    x = np.linspace(0, 1, 100)

    for n in range(len(predict_list)):
        predictions = predict_list[n]
        targets = targ_list[n]
        channel = channel_list[n]

        true_values = targets[0].astype(int)
        probabilities = predictions[0].astype(float)

        fpr, tpr, thresholds = roc_curve(true_values, probabilities)
        auc = roc_auc_score(true_values, probabilities)

        ax.plot(fpr, tpr, label=f'{channel} AUC = {auc:0.4f}')

    ax.plot(x, x, linestyle='--', label='Random AUC = 0.5')
    ax.minorticks_on()
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    plt.legend(prop={'size': 6})
    plt.savefig(f'{directory}/roc_auc_channels_variation.png', dpi=600)
    plt.clf()
    plt.close()


p, t, e, be, c = [], [], [], [], []
files = file_list()

print(f'Plotting for {len(files)} files')

for j in range(files.shape[0]):
    pickle_file = open(f'{directory}/{files[j,0]}', 'rb')
    data_info = pickle.load(pickle_file)

    data_predictions = np.load(f'{directory}/{files[j,1]}')  # Predictions data
    predictions, targets = data_predictions['predictions'], data_predictions['targets']

    p.append(predictions)
    t.append(targets)
    e.append(data_info['Epochs'])
    #be.append(data_info['Best Value Epoch'])
    #c.append(data_info['Channels'])  # For ECCConvL
    c.append(data_info['Dimension Removed'])
    be.append(data_info['Dimension Removed'])
    data_training = np.load(f'{directory}/{files[j,2]}')  # History of learning data
    loss_values, val_loss_values, accuracy_values, val_accuracy_values = data_training['loss_values'], \
                                                                         data_training['val_loss_values'], \
                                                                         data_training['accuracy_values'], \
                                                                         data_training['val_accuracy_values']

    # Individual runs plotting

    plot_history(loss_values, val_loss_values, accuracy_values, val_accuracy_values,
                 int(data_info['Best Value Epoch']), data_info['t0'])
    plot_sig_back_hist(predictions, targets, data_info['t0'])
    plot_sig_back_cumulative(predictions, targets, data_info['t0'])
    # plot_roc_curve_plt(predictions, targets, data_info['t0'])
    plot_sig_sqrt_back(predictions, targets, data_info['t0'])
    plot_roc_curve_skl(predictions, targets, data_info['t0'])

# Combining all runs plotting

plot_roc_auc_epoch_variation(p, t, e, be)  # SMPL results
#plot_roc_auc_channels(p, t, c)  # ECCConv results

print('Plotting Complete')
