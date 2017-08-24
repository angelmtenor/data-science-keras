"""
Helper module for Data-Science-Keras repository
"""
import os
import random as rn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import seaborn as sns

def remove_lowfreq(df, freq=0.01):
    """
    Remove low frequency values appearing less than 'freq' in its column of the dataframe 'df'
    """
    threshold = df.shape[0]*freq

    for f in df:
        count = df[f].value_counts()  
        low_freq = list(count[count < threshold].index)
        if len(low_freq) > 0:
            df.loc[:,f] = df.loc[:,f].replace(low_freq, np.nan)
            #df.loc[:,f] = df.loc[:,f].replace(np.nan,np.nan) 
        print(f, dict(df[f].value_counts()))
    return df


def show_missing(df, figsize=(8,3)):
    """ Display barplot with the ratio of missing values (NaN) for each column of the dataset """
    plt.figure(figsize=figsize)
    plt.ylim([0, 1])
    plt.title("Missing values")
    plt.ylabel("Missing / Total")
    (df.isnull().sum()/df.shape[0]).plot.bar()


def show_numerical(df, numerical, target=[], kde=False, figsize=(17,4)):
    """
    Display histograms of numerical features
    If a target list is provided, their histograms will be excluded
    """
    numerical_f = [n for n in numerical if n not in target]
    fig, ax = plt.subplots(ncols=len(numerical_f), sharey=False, figsize=[17,2])
    for idx, n in enumerate(numerical_f):
        sns.distplot(df[n].dropna(), ax=ax[idx], kde=kde)        
#         for value in df_filtered[t].unique():           
#             sns.distplot(df.loc[df_filtered[t]==value, n].dropna(), ax=ax[idx])
#             plt.legend(df_filtered[t].unique(), title=t)
#             # ax[idx].yaxis.set_visible(False)


def show_target_vs_categorical(df, target, categorical, figsize=(17,4)):
    """ Display barplots of target vs categorical variables
    input: pandas dataframe, target list, categorical features list
    Target values must be numerical for barplots
    """
    categorical_f = [c for c in categorical if c not in target]
    
    for t in target:   # in case of several targets several plots will be shown
        fig, ax = plt.subplots(ncols=len(categorical_f), sharey=True, figsize=figsize)

        for idx, f in enumerate(categorical_f):
            v = [v for v in df[f].values if str(v) != 'nan']
            sorted_values = sorted(set(v))
            sns.barplot(data=df, x=f, y=t, ax=ax[idx])


def show_target_vs_numerical(df, target, numerical, jitter=0, figsize=(17,4)):
    """ Display histograms of binary target vs numerical variables
    input: pandas dataframe, target list, numerical features list
    Target values must be numerical
    """
    numerical_f = [n for n in numerical if n not in target]
    
    for t in target:   # in case of several targets several plots will be shown
        fig, ax = plt.subplots(ncols=len(numerical_f), sharey=True, figsize=figsize)

        for idx, f in enumerate(numerical_f):
            g = sns.regplot(x=f, y=t, data=df, x_jitter=jitter, y_jitter=jitter, ax=ax[idx], marker=".")


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
