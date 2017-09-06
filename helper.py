"""
Helper module for Data-Science-Keras repository
"""
import os
import random as rn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf


def classify_data(df, target, numerical=None, categorical=None):
    """  Return a new dataframe with categorical variables as dtype 'categorical' and sorted
    columns: numerical + categorical + target.
    
    Input: dataframe, target list, numerical list, categorical list
    Output: classified and sorted dataframe
    """

    df = df.copy()

    assert numerical or categorical, "Numerical or categorical variable list must be provided"

    if not categorical:
        categorical = [col for col in df if col not in numerical]
    if not numerical:
        numerical = [col for col in df if col not in categorical]

    numerical_f = [col for col in numerical if col not in target]
    categorical_f = [col for col in categorical if col not in target]

    # sort columns of dataframe
    df = df[numerical_f + categorical_f + target]

    # assign category data type to categorical columns
    for f in df[categorical]:
        df[f] = df[f].astype('category')

    return df


def remove_lowfreq(df, freq=0.01, show=False, inplace=False):
    """
    Remove low frequency values appearing less than 'freq' in its column of the dataframe 'df'
    """

    if not inplace:
        df = df.copy()

    threshold = df.shape[0] * freq

    for f in df:
        count = df[f].value_counts()
        low_freq = list(count[count < threshold].index)
        if len(low_freq) > 0:
            df.loc[:, f] = df.loc[:, f].replace(low_freq, np.nan)
            # df.loc[:,f] = df.loc[:,f].replace(np.nan,np.nan)
        if show:
            print(f, dict(df[f].value_counts()))

    if not inplace:
        return df


def remove_outliers(df, sigma=3, inplace=False):
    """
    Remove outliers from numerical variables
    """
    if not inplace:
        df.copy()

    num_df = df.select_dtypes(include=[np.number])
    # col = list(num_df)
    df[num_df.columns] = num_df[np.abs(num_df - num_df.mean()) <= (sigma * num_df.std())]
    print(list(num_df))

    if not inplace:
        return df


def missing(df, limit=None, figsize=None, plot=True):
    """ 
    Display the ratio of missing values (NaN) for each column of df
    Only columns with missing values are shown
    If limit_ratio is provided, return column names exceeding the ratio (features with little data)
    """

    size = df.shape[0]
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    missing = missing.sort_values(ascending=True)
    missing_ratio = missing / size

    if plot:
        if not figsize:
            figsize=(8,missing_ratio.shape[0]//2+1)
        plt.figure(figsize=figsize)
        plt.ylim([0, 1])
        plt.xlabel("Missing / Total")
        missing_ratio.plot(kind='barh')
        if limit:
            plt.axvline(limit, linestyle='--', color='k')

    if limit:
        return missing_ratio[missing_ratio>limit].index.tolist()


def show_numerical(df, target=None, kde=False, sharey=False, figsize=(17, 2)):
    """
    Display histograms of numerical features
    If a target list is provided, their histograms will be excluded
    """
    if target is None:
        target = []

    numerical = list(df.select_dtypes(include=[np.number]))

    numerical_f = [n for n in numerical if n not in target]
    fig, ax = plt.subplots(ncols=len(numerical_f), sharey=sharey, figsize=figsize)
    for idx, n in enumerate(numerical_f):
        sns.distplot(df[n].dropna(), ax=ax[idx], kde=kde)
        #         for value in df_filtered[t].unique():


def show_target_vs_numerical(df, target, jitter=0, figsize=(17, 4)):
    """ Display histograms of binary target vs numerical variables
    input: pandas dataframe, target list 
        Target values should be numerical
    """

    numerical = list(df.select_dtypes(include=[np.number]))

    numerical_f = [n for n in numerical if n not in target]
    copy_df = df.copy()
    for t in target:
        if copy_df[t].dtype.name == 'category':
            copy_df[t] = copy_df[t].astype(int)

    for t in target:  # in case of several targets several plots will be shown
        fig, ax = plt.subplots(ncols=len(numerical_f), sharey=True, figsize=figsize)

        for idx, f in enumerate(numerical_f):
            axs = sns.regplot(x=f, y=t, data=copy_df, x_jitter=jitter, y_jitter=jitter, ax=ax[idx], marker=".")
            # first y-axis label only
            if idx != 0:
                axs.set(ylabel='')


def show_categorical(df, target=None, sharey=False, figsize=(17, 2)):
    """
    Display histograms of categorical features
    If a target list is provided, their histograms will be excluded
    """
    if target is None:
        target = []

    categorical = list(df.select_dtypes(include=['category']))

    categorical_f = [n for n in categorical if n not in target]
    fig, ax = plt.subplots(ncols=len(categorical_f), sharey=sharey, figsize=figsize)
    for idx, n in enumerate(categorical_f):
        so = sorted({v for v in df[n].values if str(v) != 'nan'})
        axs = sns.countplot(df[n].dropna(), ax=ax[idx], order=so)
        # first y-axis label only
        if idx != 0:
            axs.set(ylabel='')


def show_target_vs_categorical(df, target, figsize=(17, 4)):
    """ 
    Display barplots of target vs categorical variables
    input: pandas dataframe, target list
    Target values must be numerical for barplots
    """

    categorical = list(df.select_dtypes(include=['category']))

    categorical_f = [c for c in categorical if c not in target]
    copy_df = df.copy()
    for t in target:
        copy_df = copy_df[pd.notnull(copy_df[t])]
        if copy_df[t].dtype.name == 'category':
            copy_df[t] = copy_df[t].astype(int)

    for t in target:  # in case of several targets several plots will be shown
        fig, ax = plt.subplots(ncols=len(categorical_f), sharey=True, figsize=figsize)
        
        for idx, f in enumerate(categorical_f):
            so = sorted({v for v in copy_df[f].values if str(v) != 'nan'})
            axs = sns.barplot(data=copy_df, x=f, y=t, ax=ax[idx], order=so)
            # first y-axis label only
            if idx != 0:
                axs.set(ylabel='') 


def show_correlation(df, target):
    """ 
    Display Pearson correlation coefficient between target and numerical features
    """

    numerical = list(df.select_dtypes(include=[np.number]))

    numerical_f = [n for n in numerical if n not in target]
    copy_df = df.copy()
    for t in target:
        if copy_df[t].dtype.name == 'category':
            copy_df[t] = copy_df[t].astype(int)

    corr = copy_df.corr().loc[numerical_f, target]
    corr.plot.bar(figsize=(8, 3))
    plt.axhline(y=0, color='k', linestyle='--', )
    plt.xlabel('feature')
    plt.ylabel('Pearson correlation coefficient')
    # sns.heatmap(corr, cmap="bwr")


def normalize(data, use_scale=None):
    """
    Normalize numerical variables (mean=0, std=1)
    
    Input: dataframe to normalize, dict(numerical_feature: [mean, std]) for use a preexistent scale 
    Output:  normalized dataframe, dict(numerical_feature: [mean, std] d   
    """
    numerical = list(data.select_dtypes(include=[np.number]))

    scale = {} if not use_scale else use_scale

    for f in numerical:
        if not use_scale:
            mean, std = data[f].mean(), data[f].std()
            data[f] = (data[f] - mean) / std
            scale[f] = [mean, std]
        else:
            data.loc[:, f] = (data[f] - scale[f][0]) / scale[f][1]
    return data, scale


def create_dummy(data, target, use_dummies=None):
    """ 
    Replace categorical features by dummy features (no target)  
    If no dummy list is used, a new one is created.  
    
    Input: dataframe, target list, dummy list
    Output: dataframe with categorical replaced by dummies, generated dummy list
     """

    dummies = []

    for f in list(data.select_dtypes(include=['category'])):
        if f not in target:
            dummy = pd.get_dummies(data[f], prefix=f, drop_first=False)
            data = pd.concat([data, dummy], axis=1)
            data.drop(f, axis=1, inplace=True)

            dummies.extend(dummy)

    if use_dummies:
        missing = set(use_dummies) - set(dummies)
        for m in missing:
            data[m] = 0

    # set new columns to category
    for dummy in dummies:
        data[dummy] = data[dummy].astype('category')

    return data, dummies


def reproducible(seed=42):
    import keras
    """ Setup reproducible results from run to run using Keras
    https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    """

    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)
    rn.seed(seed)
    # Multiple threads are a potential source of non-reproducible results.
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.set_random_seed(seed)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    keras.backend.set_session(sess)


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
        print("\nTraining accuracy: \t{:.3f}".format(hist['acc'][-1]))
    if 'val_acc' in hist:
        print("Validation accuracy:\t{:.3f}".format(hist['val_acc'][-1]))


def expand_date(timeseries):
    """
    Expand a pandas datetime series returning a dataframe with these columns:
    - hour : 0 - 23
    - year:
    - month: 1 - 12
    - weekday : 0 Monday - 6 Sunday
    - holiday : 0 - 1 holiday
    - workingday : 0 weekend or holiday - 1 workingday

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


def ml_models(x_train, y_train, x_test, y_test, cross_validation=False):
    """
    Train with classical machine learning models and show the test results
    if cross_validation=True an additional training with cross validation will be performed
    """
    from time import time
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
    from sklearn.metrics import accuracy_score

    from sklearn.model_selection import KFold
    from sklearn.base import clone

    classifiers = (GaussianNB(), SVC(kernel="rbf", ), DecisionTreeClassifier(),
                   KNeighborsClassifier(n_neighbors=10), AdaBoostClassifier(), RandomForestClassifier(100))

    names = ["Naive Bayes", "SVM", "Decision Trees", "KNeighbors", "AdaBoost", "Random Forest"]

    for idx, clf in enumerate(classifiers):

        clf_cv = clone(clf)

        print("\n", names[idx], "\n", "-" * 20)

        t0 = time()
        # Fitting the model without cross validation
        clf.fit(x_train, y_train[:, 0])
        train_time = time() - t0
        y_pred = clf.predict(x_test)
        accuracy = accuracy_score(y_pred, y_test[:, 0])

        if cross_validation:
            k_fold = KFold(n_splits=10)

            t0 = time()
            # Fitting the model with cross validation
            for id_train, id_test in k_fold.split(x_train):
                # print(y_train[id_train, 0].shape)
                clf_cv.fit(x_train[id_train], y_train[id_train, 0])
            train_time_cv = time() - t0

            y_pred_cv = clf_cv.predict(x_test)
            accuracy_cv = accuracy_score(y_pred_cv, y_test[:, 0])

        print("Test Accuracy:  \t {:.3f}".format(accuracy))
        if cross_validation:
            print("Test Accuracy CV:\t {:.3f}".format(accuracy_cv))

        print("Training Time:  \t {:.1f} ms".format(train_time * 1000))
        if cross_validation:
            print("Training Time CV: \t {:.1f} ms".format(train_time_cv * 1000))


''' # test expand_date:
df = pd.DataFrame()
dates = [
    '2016-12-25 17:24:55', '2016-06-12 00:43:35', '2016-01-19 11:35:24',
    '2016-04-06 19:32:31', '2016-02-15 13:30:55']

df['Dates'] = pd.to_datetime(dates)

sample = expand_date(df['Dates'])
print(sample.head()) '''

