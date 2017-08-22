"""
Helper module for Data-Science-Keras repository
"""
import os
import random as rn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

def show_missing(df):
    """ Display barplot with the ratio of missing values (NaN) for each column of the dataset """

    plt.ylim([0, 1])
    plt.title("Missing values")
    plt.ylabel("Missing / Total")
    (df.isnull().sum()/df.shape[0]).plot.bar()


def reproducible(seed=42):
    """ Setup reproducible results from run to run using Keras
    https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    """
    from keras import backend as K

    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)
    rn.seed(seed)
    # Multiple threads are a potential source of non-reproducible results.
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.set_random_seed(seed)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)


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
        
    plt.show()    

    # show final results

    print("\nTraining loss:  \t{:.4f}".format(hist['loss'][-1]))
    if 'val_loss' in hist:
        print("Validation loss: \t{:.4f}".format(hist['val_loss'][-1]))
    if 'acc' in hist:
        print("\nTraining accuracy: \t{:.2f}".format(hist['acc'][-1]))
    if 'val_acc' in hist:
        print("Validation accuracy:\t{:.2f}".format(hist['val_acc'][-1]))




def expand_date(timeseries):
    """
    Expand a pandas datetime series returning a dataframe with these columns:
	- hour : 0 - 23
	- year: 
	- month: 1 - 12
  	- weekday : 0 Monday - 6 Sunday
	- holiday : 0 - 1 holiday
    - workingday : 0 weekend or holiday - 1 workingday ,

    """
    from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

    assert type(
        timeseries) == pd.core.series.Series, 'input must be pandas series'
    assert timeseries.dtypes == 'datetime64[ns]', 'input must be pandas datetime'

    df = pd.DataFrame()

    df['hour'] = timeseries.dt.hour  

    date = timeseries.dt.date
    df['year'] = pd.DatetimeIndex(date).year
    df['month'] = pd.DatetimeIndex(date).month
    df['day'] = pd.DatetimeIndex(date).day
    df['weekday'] = pd.DatetimeIndex(date).weekday

    holidays = calendar().holidays(start=date.min(), end=date.max())
    hol = date.astype('datetime64[ns]').isin(holidays)
    df['holiday'] = hol.values.astype(int)
    df['workingday'] = ((df['weekday'] < 5) & (df['holiday'] == 0)).astype(int)

    return df


''' # test expand_date:
df = pd.DataFrame()
dates = [
    '2016-12-25 17:24:55', '2016-06-12 00:43:35', '2016-01-19 11:35:24',
    '2016-04-06 19:32:31', '2016-02-15 13:30:55']

df['Dates'] = pd.to_datetime(dates)

sample = expand_date(df['Dates'])
print(sample.head()) '''
