import matplotlib.pyplot as plt
import seaborn as sns


def plot_training(history):
    plt.figure(figsize=(14, 4))
    plt.subplot(121)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.subplot(122)
    plt.plot(history.history['acc'], label='Training')
    plt.plot(history.history['val_acc'], label='Validation')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
