from typing import List
import tensorflow as tf
import matplotlib.pyplot as plt


def plot_training_data(history: tf.keras.callbacks.History, fig_path: str):
    fig, axs = plt.subplots(nrows=2, ncols=1)
    max_f1 = max(history['f1_score'])
    max_val_f1 = max(history['val_f1_score'])

    axs[0].plot(history['f1_score'], 'b')
    axs[0].plot(history['val_f1_score'], 'r')
    axs[0].set_title('F1 Score')
    axs[0].set_ylim(bottom=0.5, top=1)
    axs[0].annotate('max f1 {:.2f}\nmax val f1 {:.2f}'.format(
        max_f1, max_val_f1),
                    xy=(.75, .75),
                    xycoords='axes fraction')

    axs[1].plot(history['loss'], 'b')
    axs[1].plot(history['val_loss'], 'r')
    axs[1].set_title('Loss')
    axs[1].set_ylim(bottom=0, top=1)

    plt.tight_layout()
    plt.savefig(fig_path)
    return


def plot_k_fold_data(histories: List[tf.keras.callbacks.History],
                     fig_path: str):
    fig, axs = plt.subplots(nrows=4, ncols=1)
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    f1_scores = []
    val_f1_scores = []
    for i, history in enumerate(histories):
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        f1_score = history.history['f1_score']
        val_f1_score = history.history['val_f1_score']
        min_val_loss = min(val_loss)
        max_val_f1_score = max(val_f1_score)
        f1_scores.append(max(f1_score))
        val_f1_scores.append(max(val_f1_score))

        axs[0].plot(loss, colors[i])
        axs[0].set_title('Loss')
        axs[0].set_ylim(bottom=0, top=1)
        axs[1].plot(val_loss, colors[i])
        axs[1].set_title('Val Loss')
        axs[1].set_ylim(bottom=0, top=1)
        axs[2].plot(f1_score, colors[i])
        axs[2].set_title('F1 Score')
        axs[2].set_ylim(bottom=0.5, top=1)
        axs[3].plot(val_f1_score, colors[i])
        axs[3].set_title('Val F1 Score')
        axs[3].set_ylim(bottom=0.5, top=1)
    f1_strings = ' '.join('{:.2f}'.format(f1) for f1 in f1_scores)
    val_f1_strings = ' '.join('{:.2f}'.format(f1) for f1 in val_f1_scores)
    axs[2].annotate('max f1 scores: {}\nmax val f1 scores: {}'.format(
        f1_strings, val_f1_strings),
                    xy=(.75, .75),
                    xycoords='axes fraction')
    plt.tight_layout()
    plt.savefig(fig_path)
    return