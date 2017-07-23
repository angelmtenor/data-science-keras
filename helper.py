import matplotlib.pyplot as plt
import seaborn as sns


def plot_training(history):
    plt.figure(figsize=(14, 4))
    plt.subplot(121)
    plt.plot(history.history['loss'], label='Training')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    if 'acc' in history.history:
        plt.subplot(122)
        plt.plot(history.history['acc'], label='Training')
        if 'val_acc' in history.history:
            plt.plot(history.history['val_acc'], label='Validation')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()
