"""
Helper module for Data-Science-Keras repository
"""
import os, warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

from time import time
import random as rn
import math


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set()  # set seaborn style


def info_gpu():
    """ Show GPU device (if available), keras version and tensorflow version """
    import tensorflow as tf
    import keras

    # Check for a GPU
    if not tf.test.gpu_device_name():
        print('-- No GPU  --')
    else:
        print('{}'.format(tf.test.gpu_device_name()))

    # Check TensorFlow Version
    print('Keras\t\tv{}'.format(keras.__version__))
    print('TensorFlow\tv{}'.format(tf.__version__))


def reproducible(seed=42):
    import tensorflow as tf
    import keras
    """ Setup reproducible results from run to run using Keras
    https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    """

    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)
    rn.seed(seed)
    # Multiple threads are a potential source of non-reproducible results.
    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.set_random_seed(seed)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    keras.backend.set_session(sess)


# DATA PROCESSING ------------------------------------------------


def force_categorical(df):
    """ Force non numerical fields to pandas 'category' type """
    non_numerical = list(df.select_dtypes(exclude=[np.number]))

    fields_to_change = [
        f for f in df if f in non_numerical and df[f].dtype.name != 'category'
    ]

    for f in fields_to_change:
        df[f] = df[f].astype('category')

    if fields_to_change:
        print("Non-numerical fields changed to 'category':", fields_to_change)

    return df


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

    # assign float data type to numerical columns
    for n in df[numerical]:
        df[n] = df[n].astype(np.float32)

    # assign category data type to categorical columns (force_categorical not needed)
    for f in df[categorical]:
        df[f] = df[f].astype('category')

    print('Numerical features: \t{}'.format(len(numerical_f)))
    print('Categorical features: \t{}'.format(len(categorical_f)))
    for t in target:
        print("Target: \t\t{} ({})".format(t, df[t].dtype))
    return df


def remove_lowfreq(df, target=None, ratio=0.01, show=False, inplace=False):
    """
    Remove low frequency categorical values appearing less than 'ratio' in its column of the dataframe 'df'
    Only non-numerical columns are evaluated
    """

    warnings.warn(
        ' Use new "remove_categories" function',
        DeprecationWarning,
        stacklevel=2)

    if not inplace:
        df = df.copy()

    threshold = df.shape[0] * ratio

    if not target:
        target = []

    df = force_categorical(df)
    categorical = df.select_dtypes(include=['category'])
    categorical_f = [c for c in categorical if c not in target]

    if not categorical_f:
        print('None categorical variables found')

    for f in categorical_f:

        count = df[f].value_counts()
        low_freq = list(count[count < threshold].index)
        if len(low_freq) > 0:
            df[f] = df[f].replace(low_freq, np.nan)
            df[f].cat.remove_unused_categories(inplace=True)
            # df.loc[:,f] = df.loc[:,f].replace(np.low_freq, np.nan)
        if show:
            print(f, dict(df[f].value_counts()))

    if not inplace:
        return df


def remove_categories(df,
                      target=None,
                      ratio=0.01,
                      show=False,
                      dict_categories=None):
    """
    Remove low frequency categorical values appearing less than 'ratio' in its column of the dataframe 'df'
    Only non-numerical columns are evaluated
    Return: modified df, dict_categories
    """

    df = df.copy()

    threshold = df.shape[0] * ratio

    if not target:
        target = []

    df = force_categorical(df)
    categorical = df.select_dtypes(include=['category'])
    categorical_f = [c for c in categorical if c not in target]

    if not categorical_f:
        print('None categorical variables found')

    if dict_categories:
        for f in categorical_f:
            df[f].cat.set_categories(dict_categories[f], inplace=True)

    else:
        dict_categories = dict()

        for f in categorical_f:

            count = df[f].value_counts()
            low_freq = list(count[count < threshold].index)
            if low_freq:
                df[f] = df[f].replace(low_freq, np.nan)
                df[f].cat.remove_unused_categories(inplace=True)
                # df.loc[:,f] = df.loc[:,f].replace(np.low_freq, np.nan)

            dict_categories[f] = df[f].cat.categories

            if show:
                print(f, list(df[f].value_counts()))

    return df, dict_categories


def remove_outliers(df, sigma=3, inplace=False):
    """
    Remove outliers from numerical variables
    """
    if not inplace:
        df = df.copy()

    num_df = df.select_dtypes(include=[np.number])
    # col = list(num_df)
    df[num_df.columns] = num_df[np.abs(num_df - num_df.mean()) <=
                                (sigma * num_df.std())]
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
    m = df.isnull().sum()
    m = m[m > 0]
    if m.empty:
        print("No missing values found")
        return []

    m = m.sort_values(ascending=True)
    missing_ratio = m / size

    if plot:
        if not figsize:
            figsize = (8, missing_ratio.shape[0] // 2 + 1)
        plt.figure(figsize=figsize)
        plt.xlim([0, 1])
        plt.xlabel("Missing / Total")
        missing_ratio.plot(kind='barh')
        if limit:
            plt.axvline(limit, linestyle='--', color='k')

    if limit:
        return missing_ratio[missing_ratio > limit].index.tolist()


def simple_fill(df,
                target,
                include_numerical=True,
                include_categorical=True,
                inplace=False):
    warnings.warn('Use new "fill_simple" function', stacklevel=2)

    return fill_simple(
        df,
        target,
        include_numerical=include_numerical,
        include_categorical=include_categorical,
        inplace=inplace)


def fill_simple(df,
                target,
                missing_numerical='median',
                missing_categorical='mode',
                include_numerical=True,
                include_categorical=True,
                inplace=False):
    """
    Fill missing numerical values of df with the median of the column ((include_numerical=True)
    Fill missing categorical values of df with the median of the column (include_categorical=True)
    Target column is not evaluated
    """

    if not inplace:
        df = df.copy()

    numerical = list(df.select_dtypes(include=[np.number]))
    numerical_f = [col for col in numerical if col not in target]
    categorical_f = [
        col for col in df if col not in numerical and col not in target
    ]

    # numerical

    if include_numerical:
        for f in numerical_f:
            if missing_numerical == 'median':
                df[f].fillna(df[f].median(), inplace=True)
            elif missing_numerical == 'mean':
                df[f].fillna(df[f].mean(), inplace=True)
            else:
                warnings.warn("missing_numerical must be 'mean' or 'median'")
                print('Missing numerical filled with: {}'.format(
                    missing_numerical))

    # categorical

    if include_categorical:

        if missing_categorical == 'mode':
            modes = df[categorical_f].mode()
            for idx, f in enumerate(df[categorical_f]):
                df[f].fillna(modes.iloc[0, idx], inplace=True)
        else:
            for f in categorical_f:
                if missing_categorical not in df[f].cat.categories:
                    df[f].cat.add_categories(missing_categorical, inplace=True)
                df[f].fillna(missing_categorical, inplace=True)
            print('Missing categorical filled with label: "{}"'.format(
                missing_categorical))

    #df[categorical_f].apply(lambda x:x.fillna(x.value_counts().index[0], inplace=True))

    if not inplace:
        return df


# DATA EXPLORATION ------------------------------------------------

def is_binary(data):
    """ Return True if the input series (column of dataframe) is binary """
    return len(data.squeeze().unique()) == 2


def info_data(df, target=None):
    """ Display basic information of the dataset and the target (if provided) """
    n_samples = df.shape[0]
    n_features = df.shape[1] - len(target)
        
    print("Samples: \t{} \nFeatures: \t{}".format(n_samples, n_features))  

    if target:
        print("Target: \t{}".format(target))  

        if is_binary(df[target]):
            counts = (df[target].squeeze().value_counts(dropna=False))
            print("\nBinary target: \t{}".format(counts.to_dict()))
            print("Ratio \t\t{:.1f} : {:.1f}".format(counts[0]/min(counts), counts[1]/min(counts)))
            print("Dummy accuracy:\t{:.2f}".format(max(counts)/sum(counts)))


def get_types(df):
    """ Return a dataframe with the types of dataframe df """
    return pd.DataFrame(dict(df.dtypes), index=["Type"])[df.columns]


def show_numerical(df, target=None, kde=False, sharey=False, figsize=(17, 2), ncols=5):
    """
    Display histograms of numerical features
    If a target list is provided, their histograms will be excluded
    """
    if ncols <= 1:
        ncols =5
        print( "Number of columns changed to {}".format(ncols))

    if target is None:
        target = []

    numerical = list(df.select_dtypes(include=[np.number]))
    numerical_f = [n for n in numerical if n not in target]

    if not numerical_f:
        print("There are no numerical features")
        return

    nrows = math.ceil(len(numerical_f)/ncols)

    for row in range(nrows):

        if row == nrows-1 and len(numerical_f) % ncols == 1: # case 1 only plot in last row
            plt.subplots(ncols=1, figsize=figsize)
            sns.distplot(df[numerical_f[-1]].dropna(), kde=kde)
   
        else: # standard case
    
            if row == nrows-1 and len(numerical_f) % ncols != 0:  
                ncols =   len(numerical_f) % ncols # adjust size of last row  
                
            _, ax = plt.subplots(ncols=ncols, sharey=sharey, figsize=figsize)

            for idx, n in enumerate(numerical_f[row * ncols : row * ncols + ncols]):
                sns.distplot(df[n].dropna(), ax=ax[idx], kde=kde)


def show_target_vs_numerical(df,
                             target,
                             jitter=0,
                             fit_reg=True,
                             point_size=1,
                             figsize=(17, 4),
                             ncols=5):
    """ Display histograms of binary target vs numerical variables
    input: pandas dataframe, target list 
        Target values must be parsed to numbers
    """

    numerical = list(df.select_dtypes(include=[np.number]))
    numerical_f = [n for n in numerical if n not in target]

    if ncols <= 1:
        ncols =5
        print( "Number of columns changed to {}".format(ncols))
 
    if not numerical_f:
        print("There are no numerical features")
        return

    df = df.copy()

    for t in target:
        if t not in numerical:
            df[t] = df[t].astype(
                np.float16
            )  # force categorical values to numerical (booleans, ...)

    nrows = math.ceil(len(numerical_f)/ncols)

    for t in target:  # in case of several targets several plots will be shown

        for row in range(nrows):

            if row == nrows-1 and len(numerical_f) % ncols == 1: # case 1 only plot in last row
                plt.subplots(ncols=1, figsize=figsize)

                axs = sns.regplot(
                    x=df[numerical_f[-1]],
                    y=t,
                    data=df,
                    x_jitter=jitter,
                    y_jitter=jitter,
                    marker=".",
                    scatter_kws={'s': point_size * 2},
                    fit_reg=fit_reg)

            else:

                if row == nrows-1 and len(numerical_f) % ncols != 0:  
                    ncols =   len(numerical_f) % ncols # adjust size of last row  

                _, ax = plt.subplots(ncols=ncols, sharey=True, figsize=figsize)

                for idx, f in enumerate(numerical_f[row * ncols : row * ncols + ncols]):
                    axs = sns.regplot(
                        x=f,
                        y=t,
                        data=df,
                        x_jitter=jitter,
                        y_jitter=jitter,
                        ax=ax[idx],
                        marker=".",
                        scatter_kws={'s': point_size},
                        fit_reg=fit_reg)
                        
                    # first y-axis label only
                if idx != 0:
                    axs.set(ylabel='')
                    
                if is_binary(df[target]):
                    plt.ylim(ymin=-0.2, ymax=1.2)



def show_categorical(df, target=None, sharey=False, figsize=(17, 2)):
    """
    Display histograms of categorical features
    If a target list is provided, their histograms will be excluded
    """
    if target is None:
        target = []

    numerical = list(df.select_dtypes(include=[np.number]))
    categorical_f = [
        col for col in df if col not in numerical and col not in target
    ]

    if not categorical_f:
        print("There are no categorical variables")
        return

    _, ax = plt.subplots(
        ncols=len(categorical_f), sharey=sharey, figsize=figsize)
    for idx, n in enumerate(categorical_f):
        so = sorted({v for v in df[n].values if str(v) != 'nan'})
        if len(categorical_f) == 1:
            axs = sns.countplot(df[n].dropna(), ax=ax, order=so)
        else:
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

    numerical = list(df.select_dtypes(include=[np.number]))
    categorical_f = [
        col for col in df if col not in numerical and col not in target
    ]

    if not categorical_f:
        print("There are no categorical variables")
        return

    copy_df = df.copy()
    for t in target:
        copy_df = copy_df[pd.notnull(copy_df[t])]
        if t not in numerical:
            copy_df[t] = copy_df[t].astype(np.float16)

    for t in target:  # in case of several targets several plots will be shown
        _, ax = plt.subplots(
            ncols=len(categorical_f), sharey=True, figsize=figsize)

        for idx, f in enumerate(categorical_f):
            so = sorted({v for v in copy_df[f].values if str(v) != 'nan'})

            if len(categorical_f) == 1:
                axs = sns.barplot(data=copy_df, x=f, y=t, ax=ax, order=so)
            else:
                axs = sns.barplot(data=copy_df, x=f, y=t, ax=ax[idx], order=so)

            # first y-axis label only
            if idx != 0:
                axs.set(ylabel='')


def correlation(df, target, limit=None, figsize=None, plot=True):
    """ 
    Display Pearson correlation coefficient between target and numerical features
    Return a list with low-correlated features if limit is provided

    """

    numerical = list(df.select_dtypes(include=[np.number]))
    numerical_f = [n for n in numerical if n not in target]

    if not numerical_f:
        print("There are no numerical features")
        return

    copy_df = df.copy()
    for t in target:
        if t not in numerical:
            copy_df[t] = copy_df[t].astype(np.float16)

    corr = copy_df.corr().loc[numerical_f, target].fillna(0).sort_values(target, ascending=False).round(2)

    if not figsize:
        figsize = (8, len(numerical_f) // 2 + 1) 
    corr.plot.barh(figsize=figsize)
    plt.gca().invert_yaxis()
    
    if limit:
        plt.axvline(x=-limit, color='k', linestyle='--', )
        plt.axvline(x=limit, color='k', linestyle='--', )
    plt.xlabel('feature')
    plt.ylabel('Pearson correlation coefficient')

    if limit:
        return corr.loc[abs(corr[target[0]]) < abs(limit)].index.tolist()


def show_correlation(df, target, limit=None, figsize=None):
    """ 
    Display Pearson correlation coefficient between target and numerical features
    """

    warnings.warn(
        ' Use new "scale" function', DeprecationWarning, stacklevel=2)

    return correlation(df, target, limit=limit, figsize=figsize)



# DATA PROCESSING FOR ML & DL ---------------------------------------------


def scale(data, scale_param=None, method='std'):
    """
    Standardize numerical variables (mean=0, std=1)
    
    Input: dataframe to standardize, dict(numerical_feature: [mean, std]) for use a preexistent scale 
    Output:  normal-distributed dataframe, dict(numerical_feature: [mean, std]   
    """

    assert method == 'std' or method == 'minmax' or method == 'maxabs'

    data = data.copy()

    num = list(data.select_dtypes(include=[np.number]))
    if not scale_param:
        create_scale = True
        scale_param = {}
    else:
        create_scale = False

    for f in num:
        if method == 'std':
            if create_scale:
                mean, std = data[f].mean(), data[f].std()
                data[f] = (data[f].values - mean) / std
                scale_param[f] = [mean, std]
            else:
                data.loc[:, f] = (
                    data[f] - scale_param[f][0]) / scale_param[f][1]

        elif method == 'minmax':

            if create_scale:
                min, max = data[f].min(), data[f].max()
                data[f] = (data[f].values - min) / (max - min)
                scale_param[f] = [min, max]
            else:
                min, max = scale_param[f][0], scale_param[f][1]
                data.loc[:, f] = (data[f].values - min) / (max - min)

        elif method == 'maxabs':

            if create_scale:
                min, max = data[f].min(), data[f].max()
                data[f] = 2 * (data[f].values - min) / (max - min) - 1
                scale_param[f] = [min, max]
            else:
                min, max = scale_param[f][0], scale_param[f][1]
                data.loc[:, f] = 0.5 * (data[f].values *
                                        (max - min) + max + min)

    return data, scale_param


def standardize(data, use_scale=None):
    """
    Standardize numerical variables (mean=0, std=1)
    
    Input: dataframe to standardize, dict(numerical_feature: [mean, std]) for use a preexistent scale 
    Output:  normal-distributed dataframe, dict(numerical_feature: [mean, std]   
    """
    warnings.warn(
        ' Use new "scale" function', DeprecationWarning, stacklevel=2)

    return scale(data, use_scale)


def replace_by_dummies(data, target, dummies=None, drop_first=False):
    """ 
    Replace categorical features by dummy features (no target)  
    If no dummy list is used, a new one is created.  
    
    Input: dataframe, target list, dummy list
    Output: dataframe with categorical replaced by dummies, dummy dictionary
     """

    data = data.copy()

    if not dummies:
        create_dummies = True
    else:
        create_dummies = False

    found_dummies = []

    categorical = list(data.select_dtypes(include=['category']))
    categorical_f = [col for col in categorical if col not in target]

    for f in categorical_f:
        dummy = pd.get_dummies(data[f], prefix=f, drop_first=drop_first)
        data = pd.concat([data, dummy], axis=1)
        data.drop(f, axis=1, inplace=True)

        found_dummies.extend(dummy)

    if not create_dummies:

        # remove new dummies not in given dummies
        new = set(found_dummies) - set(dummies)
        for n in new:
            data.drop(n, axis=1, inplace=True)

        # fill missing dummies with empty values (0)
        missing = set(dummies) - set(found_dummies)
        for m in missing:
            data[m] = 0

    else:
        dummies = found_dummies

    # set new columns to category
    for dummy in dummies:
        data[dummy] = data[dummy].astype('category')

    return data, dummies


def create_dummy(data, target, use_dummies=None):

    warnings.warn('Use new "replace_by_dummies" function', stacklevel=2)

    return replace_by_dummies(data, target, dummies=use_dummies)


def get_class_weight(y):
    """ Return dictionary of weight vector for imbalanced binary target """
    from sklearn.utils import class_weight

    y_plain = np.ravel(y)

    cw = class_weight.compute_class_weight('balanced',
                                           np.unique(y_plain), y_plain)

    cw = {idx: value for idx, value in enumerate(cw)}

    return cw


# MACHINE LEARNING & DEEP LEARNING ------------------------------------------------


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


def XGBClassifier(x_train,
                  y_train,
                  x_test,
                  y_test,
                  max_depth=3,
                  learning_rate=0.1,
                  n_estimators=100):
    """ Custom XGBoost classifier """

    import xgboost as xgb
    from sklearn.metrics import accuracy_score

    clf = xgb.XGBClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate)

    t0 = time()

    clf.fit(x_train, y_train)
    train_time = time() - t0
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n", "XGBoost", "\n", "-" * 20)
    print("Test Accuracy:  \t {:.3f}".format(accuracy))
    print("Training Time:  \t {:.1f} ms".format(train_time * 1000))
    return clf


def ml_classification(x_train,
                      y_train,
                      x_test,
                      y_test,
                      cross_validation=False,
                      show=False):
    """
    Build, train, and test the data set with classical machine learning classification models.
    If cross_validation=True an additional training with cross validation will be performed.
    """
    from time import time
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier

    # from sklearn.model_selection import KFold
    # from sklearn.base import clone

    classifiers = (GaussianNB(), AdaBoostClassifier(),
                   DecisionTreeClassifier(), RandomForestClassifier(100),
                   ExtraTreesClassifier(100))

    names = [
        "Naive Bayes", "AdaBoost", "Decision Tree", "Random Forest",
        "Extremely Randomized Trees"
    ]

    col = [
        'Time (s)', 'Loss', 'Accuracy', 'Precision', 'Recall', 'ROC-AUC',
        'F1-score'
    ]
    results = pd.DataFrame(columns=col)

    for idx, clf in enumerate(classifiers):

        # clf_cv = clone(clf)

        name = names[idx]
        print(name)

        t0 = time()
        # Fitting the model without cross validation

        warnings.filterwarnings(
            "ignore", message="overflow encountered in reduce")

        clf.fit(x_train, y_train)
        train_time = time() - t0
        y_pred = clf.predict_proba(x_test)

        loss, acc, pre, rec, roc, f1 = binary_classification_scores(
            y_test, y_pred[:, 1], show=show)

        if cross_validation:
            warnings.warn('Cross-validation removed')

            # k_fold = KFold(n_splits=10)

            # t0 = time()
            # # Fitting the model with cross validation
            # for id_train, id_test in k_fold.split(x_train):
            #     # print(y_train[id_train, 0].shape)
            #     clf_cv.fit(x_train[id_train], y_train[id_train, 0]) # TODO enhance
            # train_time_cv = time() - t0

            # y_pred_cv = clf_cv.predict_proba(x_test)
            # accuracy_cv = accuracy_score(y_test, y_pred_cv[:,1])
            # print("Test Accuracy CV:\t {:.3f}".format(accuracy_cv))
            # print("Training Time CV: \t {:.1f} ms".format(train_time_cv * 1000))

        results = results.append(
            pd.DataFrame(
                [[train_time, loss, acc, pre, rec, roc, f1]],
                columns=col,
                index=[name]))

    return results.sort_values('Accuracy', ascending=False).round(2)


def ml_regression(x_train,
                  y_train,
                  x_test,
                  y_test,
                  cross_validation=False,
                  show=False):
    """
    Build, train, and test the data set with classical machine learning regression models.
    If cross_validation=True an additional training with cross validation will be performed.
    """
    from time import time
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import BayesianRidge
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor

    # from sklearn.model_selection import KFold
    # from sklearn.base import clone

    regressors = (LinearRegression(), BayesianRidge(), DecisionTreeRegressor(),
                  KNeighborsRegressor(n_neighbors=10), AdaBoostRegressor(),
                  RandomForestRegressor(100))

    names = [
        "Linear", "Bayesian Ridge", "Decision Tree", "KNeighbors", "AdaBoost",
        "Random Forest"
    ]

    col = ['Time (s)', 'Test loss', 'Test R2 score']
    results = pd.DataFrame(columns=col)

    for idx, clf in enumerate(regressors):

        name = names[idx]
        # clf_cv = clone(clf)

        print(name)

        t0 = time()
        # Fitting the model without cross validation
        clf.fit(x_train, y_train)
        train_time = np.around(time() - t0, 1)
        y_pred = clf.predict(x_test)

        loss, r2 = regression_scores(y_test, y_pred, show=show)

        if cross_validation:
            warnings.warn('Cross-validation removed')

            # k_fold = KFold(n_splits=10)
            # t0 = time()
            # # Fitting the model with cross validation
            # for id_train, id_test in k_fold.split(x_train):
            #     # print(y_train[id_train, 0].shape)
            #     clf_cv.fit(x_train[id_train], y_train[id_train, 0]) # TODO enhance
            # train_time_cv = time() - t0

            # y_pred_cv = clf_cv.predict(x_test)
            # r2_cv = r2_score(y_test, y_pred_cv[:,1])

            # print("Test R2-Score CV:\t {:.3f}".format(r2_cv))
            # print( "Training Time CV: \t {:.1f} ms".format(train_time_cv * 1000))

        results = results.append(
            pd.DataFrame([[train_time, loss, r2]], columns=col, index=[name]))

        if show:
            print("-" * 20)
            print("Training Time:  \t {:.1f} s".format(train_time))
            print("Test loss:  \t\t {:.4f}".format(loss))
            print("Test R2-score:  \t {:.3f}\n".format(r2))

    return results.sort_values('Test loss')


def binary_classification_scores(y_test, y_pred, return_dataframe=False, index=" ", show=True):
    """ Return classification metrics: log_loss, acc, precision, recall, roc_auc, F1 score """

    from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score
    from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

    rec, roc, f1 = 0, 0, 0

    y_pred_b = (y_pred > 0.5).astype(int)

    warnings.filterwarnings(
        "ignore", message="divide by zero encountered in log")
    warnings.filterwarnings(
        "ignore", message="invalid value encountered in multiply")
    loss = log_loss(y_test, y_pred)

    acc = accuracy_score(y_test, y_pred_b)

    warnings.filterwarnings(
        "ignore",
        message=
        "Precision is ill-defined and being set to 0.0 due to no predicted samples"
    )
    pre = precision_score(y_test, y_pred_b)

    if acc > 0 and pre > 0:
        rec = recall_score(y_test, y_pred_b)
        roc = roc_auc_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred_b)

    if show:
    #     print('Scores:\n' + '-' * 11)
    #     print('Log_Loss: \t{:.4f}'.format(loss))
    #     print('Accuracy: \t{:.2f}'.format(acc))
    #     print('Precision: \t{:.2f}'.format(pre))
    #     print('Recall: \t{:.2f}'.format(rec))
    #     print('ROC AUC: \t{:.2f}'.format(roc))
    #     print('F1-score: \t{:.2f}'.format(f1))
        print('\nConfusion matrix: \n', confusion_matrix(y_test, y_pred_b))

    if return_dataframe:
        col = ['Loss', 'Accuracy', 'Precision', 'Recall', 'ROC-AUC','F1-score']
        scores = pd.DataFrame([[loss, acc, pre, rec, roc, f1]], columns=col, index=[index]).round(2)   
        return scores

    return loss, acc, pre, rec, roc, f1


def regression_scores(y_test, y_pred, show=True):
    """ Return regression metrics: (loss, R2 Score) """

    from sklearn.metrics import r2_score, mean_squared_error

    r2 = r2_score(y_test, y_pred)
    loss = mean_squared_error(y_test, y_pred)

    if show:
        print('Scores:\n' + '-' * 11)
        print('Loss (mse): \t{:.4f}'.format(loss))
        print('R2 Score: \t{:.2f}'.format(r2))

    return loss, r2


def feature_importances(features, model, top=10, plot=True):
    """ Return a dataframe with the most relevant features from a trained tree-based model """

    n = len(features) 
    if n < top:
        top = n

    #print("\n Top contributing features:\n", "-" * 26)
    importances = pd.DataFrame(data={'Importances':model.feature_importances_}, index=features)
    importances = importances.sort_values('Importances', ascending=False).round(2).head(top)

    if plot:
        figsize = (8, top // 2 + 1)
        importances.plot.barh(figsize=figsize)
        plt.gca().invert_yaxis()

    return importances
