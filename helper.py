"""
Helper module for Data-Science-Keras repository
"""

import matplotlib.pyplot as plt
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar


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

    print("\nTraining loss:  \t{:.4f}".format(hist['loss'][-1]))
    if 'val_loss' in hist:
        print("Validation loss: \t{:.4f}".format(hist['val_loss'][-1]))
    if 'acc' in hist:
        print("\nTraining accuracy: \t{:.2f}".format(hist['acc'][-1]))
    if 'val_acc' in hist:
        print("Validation accuracy:\t{:.2f}".format(hist['val_acc'][-1]))

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


def expand_date(timeseries):
    """
    Expand pandas datetime series with 'Date' returning a dataframe with these columns:
	- year : year
	- month : month (1-12)
	- hour : hour (0-23)
  	- weekday : day of the week (0 Monday - 6 Sunday)
	- holiday : weather day is holiday or not
    """

    assert type(
        timeseries) == pd.core.series.Series, 'input must be pandas series'
    assert timeseries.dtypes == 'datetime64[ns]', 'input must be pandas datetime'

    date = timeseries.dt.date

    print(date)

    df = pd.DataFrame()
    df['Year'] = pd.DatetimeIndex(date).year
    df['Month'] = pd.DatetimeIndex(date).month
    df['Day'] = pd.DatetimeIndex(date).day
    df['Weekday'] = pd.DatetimeIndex(date).weekday

    holidays = calendar().holidays(start=date.min(), end=date.max())
    df['holiday'] = date.astype('datetime64').isin(holidays)

    # df.drop('Date', axis='columns', inplace=True)
    print("Holiday", holidays)

    return df


''' 
    df = pd.DataFrame()
    dates = ['2016-12-25 17:24:55', 
            '2016-06-12 00:43:35',
            '2016-01-19 11:35:24',
            '2016-04-06 19:32:31',
            '2016-02-15 13:30:55']

    df['Dates'] =  pd.to_datetime(dates)

    sample = expand_date(df['Dates'])
    print(sample.head())
'''