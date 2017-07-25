import matplotlib.pyplot as plt


def show_training(history):
    """
    Print the final loss and plot its evolution in the training process.
    The same applies to 'validation loss', 'accuracy', and 'validation accuracy' if available
    :param history: Keras history object (model.fit return)
    :return:
    """

    hist = history.history

    if 'loss' not in hist:
        print("Error: 'loss' values not found in the history")
        return

    # show final results

    print("Training loss: {:.4f}".format(hist['loss'][-1]))
    if 'acc' in hist:
        print("Training accuracy: {:.2f}".format(hist['acc'][-1]))

    if 'val_loss' in hist:    
        print("Validation loss: {:.4f}".format(hist['val_loss'][-1]))

    if 'val_acc' in hist:
        print("Validation accuracy: {:.2f}".format(hist['val_acc'][-1]))

    # plot training

    plt.figure(figsize=(14, 4))
    plt.subplot(121)
    plt.plot(hist['loss'], label='Training')
    if 'val_loss' in hist:
        plt.plot(hist['val_loss'], label='Validation')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    if 'acc' in hist:
        plt.subplot(122)
        plt.plot(hist['acc'], label='Training')
        if 'val_acc' in hist:
            plt.plot(hist['val_acc'], label='Validation')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()
