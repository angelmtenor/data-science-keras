"""
Helper module for Data-Science-Keras repository
Angel Martinez-Tenor 2018
TODO: Update with newer functions from personal private repositories (2018-2022)
TODO: Update with best practices 2022 (ethics & green code)
TODO: Separate in different modules
"""
from __future__ import annotations

import math
import os
import platform
import random as python_random
import sys
import warnings
from pathlib import Path
from time import ctime, time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pkg_resources
import psutil
import seaborn as sns
from lightgbm import LGBMClassifier, LGBMRegressor
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

# scikit learn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (  # AdaBoostClassifier,; AdaBoostRegressor,
    ExtraTreesClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils import class_weight

# tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.models import Sequential
from tensorflow import keras

# SETUP ----------------------------------------------------------------------------------------------------------------


EXECUTION_PATH = "data-science-keras"

INSTALLED_PACKAGES = pkg_resources.working_set
installed_packages_dict = {i.key: i.version for i in INSTALLED_PACKAGES}  # pylint: disable=not-an-iterable

DEFAULT_MODULES = ("tensorflow", "pandas", "numpy")

sns.set()  # set seaborn style


def info_os() -> None:
    """Print OS version"""
    print(f"\nOS:\t{platform.platform()}")
    # print('{} {} {}'.format(platform.system(), platform.release(), platform.machine()))


def info_software(modules: list[str] = DEFAULT_MODULES) -> None:
    """Print version of Python and Python modules using pkg_resources
        note: not all modules can be obtained with pkg_resources: e.g: pytorch, mlflow ..
    Args:
        modules (list[str], optional): list of python libraries. Defaults to DEFAULT_MODULES.
    Usage Sample:
        modules = ['pandas', 'scikit-learn', 'flask', 'fastapi', 'shap', 'pycaret', 'tensorflow', 'streamlit']
        ds.info_system(hardware=True, modules=modules)
    """

    # Python Environment
    env = getattr(sys, "base_prefix", None) or getattr(sys, "real_prefix", None) or sys.prefix
    print(f"\nENV:\t{env}")

    python_version = sys.version
    print(f"\nPYTHON:\t{python_version}")

    if modules is None:
        modules = DEFAULT_MODULES

    for i in modules:
        if i in installed_packages_dict:
            print(f"{i:<25}{installed_packages_dict.get(i):>10}")
        else:
            print(f"{i:<25}: {'--NO--':>10}")
    print()


def info_hardware() -> None:
    """Print CPU, RAM, and GPU info"""

    print("\nHARDWARE:")

    # CPU INFO
    try:
        import cpuinfo  # pip py-cpuinfo

        cpu = cpuinfo.get_cpu_info().get("brand_raw")
        print(f"CPU:\t{cpu}")
    except ImportError:
        print("cpuinfo not found. (pip/conda: py-cpuinfo)")

    # RAM INFO
    ram = round(psutil.virtual_memory().total / (1024.0**3))
    print(f"RAM:\t{ram} GB")

    # GPU INFO
    if not tf.test.gpu_device_name():
        print("-- No GPU  --")
    else:
        gpu_devices = tf.config.list_physical_devices("GPU")
        details = tf.config.experimental.get_device_details(gpu_devices[0])
        gpu_name = details.get("device_name", "CUDA-GPU found")
        print(f"GPU:\t{gpu_name}")
        # print(f"{tf.test.gpu_device_name()[1:]}")


def info_system(hardware: bool = True, modules: list[str] = None) -> None:
    """Print Complete system info:
        - Show CPU & RAM hardware=True (it can take a few seconds)
        - Show OS version.
        - Show versions of Python & Python modules
        - Default list of Python modules:  ['pandas', 'scikit-learn']
    Args:
        hardware (bool, optional): Include hardware info. Defaults to True.
        modules (list[str], optional): list of python libraries. Defaults to None.
    """
    if hardware:
        info_hardware()
    info_os()
    info_software(modules=modules)

    print(f"EXECUTION PATH: {Path().absolute()}")
    print(f"EXECUTION DATE: {ctime()}")


def reproducible(seed: int = 0) -> None:
    """Setup reproducible results from run to run using Keras
    https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    Args:
        seed (int): Seed value for reproducible results. Default to 0.
    """
    os.environ["PYTHONHASHSEED"] = "0"
    np.random.seed(seed)
    python_random.seed(seed)
    tf.random.set_seed(seed)


def set_parent_execution_path(target_path: Path | str = EXECUTION_PATH) -> None:
    """Set the execution path to a parent directory (up to 3 levels). Used to execute notebooks located in a subfolder
    of the main path (e.g.: notebooks) as if they were in their parent main path (useful to reproduce prod scripts)
    Args:
        execution_path (Path or str): Execution path. e.g.: 'my-repo'. Default to EXECUTION_PATH
    TODO: Update by using environment variables with python-dotenv
    """
    target_path = Path(target_path)
    current_path = Path().absolute()

    assert str(target_path) in str(current_path), "target_path must be part of the current_path"

    if current_path.stem == target_path.stem:
        print(f"Already in target path {current_path}")
        return
    else:
        new_path = current_path
        for _ in range(3):
            new_path = new_path.parent
            if new_path.stem == target_path.stem:
                os.chdir(new_path)
                print(f"Path changed to {new_path}")
                return

    assert False, f"{target_path} not found"


# DATA PROCESSING ------------------------------------------------------------------------------------------------------


def sort_columns_by_type(df: pd.DataFrame, target: str | list = None, numerical: list = None, categorical: list = None):
    """Return a dataframe with sorted columns: numerical + categorical + target. The categorical variables are also
    converted to pandas 'category' type.
    Args:
        df (pd.DataFrame): Dataframe to sort
        target (str|list, optional): Target variable. Defaults to None.
        numerical (list, optional): List of numerical variables. Defaults to None.
        categorical (list, optional): List of categorical variables. Defaults to None.
    Returns:
        pd.DataFrame: Dataframe with sorted columns Numerical + Categorical + Target
    """

    df = df.copy()

    assert numerical or categorical, "Numerical or categorical variable list must be provided"

    if not target:
        target = []
    elif isinstance(target, str):
        target = [target]

    if not categorical:
        categorical = list({col for col in df if col not in numerical + target})
    if not numerical:
        numerical = list({col for col in df if col not in categorical + target})

    # assign float data type to numerical columns
    df[numerical] = df[numerical].astype(np.float32)

    # assign category data type to categorical columns (force_categorical not needed)
    df[categorical] = df[categorical].astype("category")

    # sort columns of dataframe
    numerical_features = [col for col in numerical if col not in target]
    categorical_features = [col for col in categorical if col not in target]

    df = df[numerical_features + categorical_features + target]
    print(f"{len(numerical_features)} numerical features: \t {numerical_features}")
    print(f"{len(categorical_features)} categorical features: \t {categorical_features}")
    for t in target:
        print(f"Target: \t\t{t} ({df[t].dtype})")
    return df


def force_categorical(df: pd.DataFrame, columns: list[str] = None) -> pd.DataFrame:
    """Force variables to pandas 'category' type. If columns is None, all non-numerical variables are converted
    Args:
        df (pd.DataFrame): Dataframe to convert
        columns (list[str], optional): List of columns to convert to categorical type. Defaults to None.
    Returns:
        pd.DataFrame: Dataframe with selected variables converted to 'category' type
    """

    if columns is None:
        columns = df.select_dtypes(exclude=["number"]).columns

    # categorical columns don't need conversion
    columns_to_convert = df[columns].select_dtypes(exclude=["category"]).columns

    for f in columns_to_convert:
        df[f] = df[f].astype("category")

    if len(columns_to_convert) > 0:
        print("Variables changed to 'category':", columns_to_convert)

    return df


def remove_categories(
    df: pd.DataFrame, target: str | list = None, ratio: float = 0.01, show: bool = False, dict_categories: dict = None
) -> tuple(pd.DataFrame, dict):
    """
    Remove low frequency categorical values appearing less than 'ratio' in its column of the input dataframe.
    Only non-numerical columns are evaluated.
    Args:
        df (pd.DataFrame): Dataframe to remove categories from
        target (str|list, optional): Target variable. Defaults to None.
        ratio (float, optional): Ratio of categories to remove. Defaults to 0.01.
        show (bool, optional): Show removed categories. Defaults to False.
        dict_categories (dict, optional): Dictionary with categories to remove. Defaults to None.
    Returns:
        pd.DataFrame: Dataframe with updated categories (after removal)
        dict: Dictionary with updated categories (key: variable, value: list of categories)
    """

    df = df.copy()
    threshold = df.shape[0] * ratio

    if not target:
        target = []
    elif isinstance(target, str):
        target = [target]

    df = force_categorical(df)
    categorical = df.select_dtypes(include=["category"])
    categorical_f = [c for c in categorical if c not in target]

    if not categorical_f:
        print("None categorical variables found")

    if dict_categories:
        for f in categorical_f:
            df[f] = df[f].cat.set_categories(dict_categories[f])

    else:
        dict_categories = dict()

        for f in categorical_f:

            count = df[f].value_counts()
            low_freq = set(count[count < threshold].index)
            # high_freq = list(count[count >= threshold].index)
            if low_freq:
                print(f"Removing {len(low_freq)} categories from feature {f}")
                df.loc[df[f].isin(low_freq), f] = np.nan

            # Slow:
            #     df[f] = df[f].replace(low_freq, np.nan)
            #     df[f] = df[f].cat.remove_unused_categories()

            # df.loc[:,f] = df.loc[:,f].replace(np.low_freq, np.nan)

            dict_categories[f] = df[f].cat.categories

            if show:
                print(f, list(df[f].value_counts()))

    return df, dict_categories


def remove_outliers(df: pd.DataFrame, sigma: float = 3) -> pd.DataFrame:
    """
    Remove outliers from numerical variables
    Args:
        df (pd.DataFrame): Dataframe to remove outliers from
        sigma (float, optional): Sigma value to remove outliers. Defaults to 3.
    Returns:
        pd.DataFrame: Dataframe with updated outliers (after removal)
    """
    df = df.copy()

    num_df = df.select_dtypes(include=[np.number])
    # col = list(num_df)
    df[num_df.columns] = num_df[np.abs(num_df - num_df.mean()) <= (sigma * num_df.std())]
    print(list(num_df))

    return df


def fill_simple(
    df: pd.DataFrame,
    target: str | list = None,
    missing_numerical: str = "median",
    missing_categorical: str = "mode",
    include_numerical: bool = True,
    include_categorical: bool = True,
) -> pd.DataFrame:
    """
    Fill missing numerical values of df with the median of the column (include_numerical=True)
    Fill missing categorical values of df with the median of the column (include_categorical=True)
    Target column, if provided, is not evaluated
    Args:
        df (pd.DataFrame): Dataframe to fill missing values of
        target (str|list, optional): Target variable. Defaults to None.
        missing_numerical (str, optional): Method to fill missing numerical values. Defaults to "median".
        missing_categorical (str, optional): Method to fill missing categorical values. Defaults to "mode".
        include_numerical (bool, optional): Include numerical columns. Defaults to True.
        include_categorical (bool, optional): Include categorical columns. Defaults to True.
    Returns:
        pd.DataFrame: Dataframe with filled missing values
    TODO: Update with scikit-learn Pipelines (from personal private repos)
    """

    df = df.copy()

    if not target:
        target = []
    elif isinstance(target, str):
        target = [target]

    numerical = list(df.select_dtypes(include=[np.number]))
    numerical_f = [col for col in numerical if col not in target]
    categorical_f = [col for col in df if col not in numerical and col not in target]

    # numerical variables

    if include_numerical:
        for f in numerical_f:
            if missing_numerical == "median":
                df[f] = df[f].fillna(df[f].median())
            elif missing_numerical == "mean":
                df[f] = df[f].fillna(df[f].mean())
            else:
                warnings.warn("missing_numerical must be 'mean' or 'median'")
                print(f"Missing numerical filled with: {missing_numerical}")

    # categorical variables

    if include_categorical:

        if missing_categorical == "mode":
            modes = df[categorical_f].mode()
            for idx, f in enumerate(df[categorical_f]):
                df[f] = df[f].fillna(modes.iloc[0, idx])
        else:
            for f in categorical_f:
                if missing_categorical not in df[f].cat.categories:
                    df[f] = df[f].cat.add_categories(missing_categorical)
                df[f] = df[f].fillna(missing_categorical)
            print(f'Missing categorical filled with label: "{missing_categorical}"')

    return df


def expand_date(timeseries: pd.Series) -> pd.DataFrame:
    """
    Expand a datetime series to a dataframe with the following columns:
    - hour : 0 - 23
    - year
    - month: 1 - 12
    - day
    - weekday : 0 Monday - 6 Sunday
    - holiday : 0 - 1 holiday (US Federal Holiday Calendar)
    - workingday : 0 weekend or holiday - 1 workingday
    Args:
        timeseries (pd.Series): Datetime series to expand
    Returns:
        pd.DataFrame: Expanded dataframe
    """

    assert isinstance(timeseries, pd.core.series.Series), "input must be pandas series"
    assert timeseries.dtypes == "datetime64[ns]", "input must be pandas datetime"

    df = pd.DataFrame()

    df["hour"] = timeseries.dt.hour

    date = timeseries.dt.dateUSFederalHolUSFederalHolidayCalendaridayCalendar
    df["year"] = pd.DatetimeIndex(date).year
    df["month"] = pd.DatetimeIndex(date).month
    df["day"] = pd.DatetimeIndex(date).day
    df["weekday"] = pd.DatetimeIndex(date).weekday

    holidays = calendar().holidays(start=date.min(), end=date.max())
    hol = date.astype("datetime64[ns]").isin(holidays)
    df["holiday"] = hol.values.astype(int)
    df["workingday"] = ((df["weekday"] < 5) & (df["holiday"] == 0)).astype(int)

    return df


# DATA EXPLORATION ----------------------------------------------------------------------------------------------------


def missing(df: pd.DataFrame, limit: float = None, figsize: tuple = None, plot: bool = True) -> None | list:
    """
    Display the ratio of missing values (NaN) for each column of df. Only columns with missing values are shown
    If limit (limit ratio) is provided, return the column names exceeding the ratio (too much missing data)
    Args:
        df (pd.DataFrame): Dataframe to evaluate
        limit (float, optional): Limit ratio of missing values. Defaults to None.
        figsize (tuple, optional): Figure size. Defaults to None.
        plot (bool, optional): Plot missing values. Defaults to True.
    Returns:
        None: If limit is not provided or list: column names exceeding the limit ratio of missing values
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
        missing_ratio.plot(kind="barh")
        if limit:
            plt.axvline(limit, linestyle="--", color="k")

    if limit:
        return missing_ratio[missing_ratio > limit].index.tolist()


def is_binary(data: pd.Series) -> bool:
    """Return True if the input series (column of dataframe) has only 2 values
    Args:
        data (pd.Series): Series to evaluate
    Returns:
        bool: True if the input has only 2 values
    """
    return len(data.squeeze().unique()) == 2


def info_data(df: pd.DataFrame, target=None) -> None:
    """Display basic information of the dataset and the target (if provided)
    Args:
        df (pd.DataFrame): Dataframe to display information from
    """
    n_samples = df.shape[0]
    n_features = df.shape[1] - len(target)

    print(f"Samples: \t{n_samples} \nFeatures: \t{n_features}")

    if target:
        for t in target:
            print(f"Target: \t{t}")
            if is_binary(df[t]):
                counts = df[t].squeeze().value_counts(dropna=False)
                print(f"Binary target: \t{counts.to_dict()}")
                print(f"Ratio \t\t{counts[0] / min(counts):.1f} : {counts[1] / min(counts):.1f}")
                print(f"Dummy accuracy:\t{max(counts) / sum(counts):.2f}")


def get_types(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe with the types of the input dataframe
    Args:
        df (pd.DataFrame): Input dataframe
    Returns:
        pd.DataFrame: Dataframe with the types of the input dataframe
    """
    return pd.DataFrame(dict(df.dtypes), index=["Type"])[df.columns]


def show_numerical(df, target=None, kde=False, sharey=False, figsize=(17, 2), ncols=5):
    """
    Display histograms of numerical features. If a target list is provided, their histograms will be excluded
    """
    if ncols <= 1:
        ncols = 5
        print(f"Number of columns changed to {ncols}")

    if target is None:
        target = []

    numerical = list(df.select_dtypes(include=[np.number]))
    numerical_f = [n for n in numerical if n not in target]

    if not numerical_f:
        print("There are no numerical features")
        return

    nrows = math.ceil(len(numerical_f) / ncols)

    for row in range(nrows):

        if row == nrows - 1 and len(numerical_f) % ncols == 1:  # case 1 only plot in last row
            plt.subplots(ncols=1, figsize=figsize)
            sns.histplot(df[numerical_f[-1]].dropna(), kde=kde)

        else:  # standard case

            if row == nrows - 1 and len(numerical_f) % ncols != 0:
                ncols = len(numerical_f) % ncols  # adjust size of last row

            _, ax = plt.subplots(ncols=ncols, sharey=sharey, figsize=figsize)

            for idx, n in enumerate(numerical_f[row * ncols : row * ncols + ncols]):  # noqa: E203
                sns.histplot(df[n].dropna(), ax=ax[idx], kde=kde)


def show_target_vs_numerical(
    df: pd.DataFrame,
    target: list[str],
    jitter: float = 0,
    fit_reg: bool = True,
    point_size: int = 1,
    figsize: tuple = (17, 4),
    ncols: int = 5,
) -> None:
    """Display scatter plots of the targets vs numerical variables.
    Args:
        df (pd.DataFrame): Input dataframe
        target (list[str]): List of target variables to display of the input dataframe
        jitter (float, optional): Jitter value for the scatter plot. Defaults to 0.
        fit_reg (bool, optional): Fit a regression line to the scatter plot. Defaults to True.
        point_size (int, optional): Size of the size of the points in the scatter plot. Defaults to 1.
        figsize (tuple, optional): Figure size. Defaults to (17, 4).
        ncols (int, optional): Max number of subplots in a row of plots. Defaults to 5.
    """

    numerical = list(df.select_dtypes(include=[np.number]))
    numerical_f = [n for n in numerical if n not in target]

    if ncols <= 1:
        ncols = 5
        print(f"Number of columns changed to {ncols}")

    if not numerical_f:
        print("There are no numerical features")
        return

    df = df.copy()

    for t in target:
        if t not in numerical:
            df[t] = df[t].astype(np.float16)  # force categorical values to numerical (booleans, ...)

    nrows = math.ceil(len(numerical_f) / ncols)

    for t in target:  # in case of several targets several plots will be shown

        for row in range(nrows):

            if row == nrows - 1 and len(numerical_f) % ncols == 1:  # case 1 only plot in last row
                plt.subplots(ncols=1, figsize=figsize)

                axs = sns.regplot(
                    x=df[numerical_f[-1]],
                    y=t,
                    data=df,
                    x_jitter=jitter,
                    y_jitter=jitter,
                    marker=".",
                    scatter_kws={"s": point_size * 2},
                    fit_reg=fit_reg,
                )

            else:

                if row == nrows - 1 and len(numerical_f) % ncols != 0:
                    ncols = len(numerical_f) % ncols  # adjust size of last row

                _, ax = plt.subplots(ncols=ncols, sharey=True, figsize=figsize)

                idx = 0
                for idx, f in enumerate(numerical_f[row * ncols : row * ncols + ncols]):  # noqa: E203
                    axs = sns.regplot(
                        x=f,
                        y=t,
                        data=df,
                        x_jitter=jitter,
                        y_jitter=jitter,
                        ax=ax[idx],
                        marker=".",
                        scatter_kws={"s": point_size},
                        fit_reg=fit_reg,
                    )

                    # first y-axis label only
                if idx != 0:
                    axs.set(ylabel="")

                if is_binary(df[t]):
                    plt.ylim(ymin=-0.2, ymax=1.2)


def show_categorical(
    df: pd.DataFrame, target: list[str] = None, sharey: bool = False, figsize: tuple = (17, 2), ncols: int = 5
) -> None:
    """
    Display histograms of categorical features. If a target list is provided, their histograms will be excluded.
    Args:
        df (pd.DataFrame): Input dataframe
        target (list[str]): List of target variables to not display. Defaults to None.
        sharey (bool, optional): Share the y-axis. Defaults to False.
        figsize (tuple, optional): Figure size. Defaults to (17, 2).
        ncols (int, optional): Max number of subplots in a row of plots. Defaults to 5.
    """
    if target is None:
        target = []

    if ncols <= 1:
        ncols = 5
        print(f"Number of columns changed to {ncols}")

    numerical = list(df.select_dtypes(include=[np.number]))
    categorical_f = [col for col in df if col not in numerical and col not in target]

    if not categorical_f:
        print("There are no categorical variables")
        return

    nrows = math.ceil(len(categorical_f) / ncols)

    for row in range(nrows):

        if row == nrows - 1 and len(categorical_f) % ncols == 1:  # case 1 only plot in last row
            plt.subplots(ncols=1, figsize=figsize)
            # so = sorted({v for v in df[nrows-1].values if str(v) != 'nan'})
            sns.countplot(x=df[categorical_f[-1]].dropna())

        else:  # standard case

            if row == nrows - 1 and len(categorical_f) % ncols != 0:
                ncols = len(categorical_f) % ncols  # adjust size of last row

            _, ax = plt.subplots(ncols=ncols, sharey=sharey, figsize=figsize)

            for idx, n in enumerate(categorical_f[row * ncols : row * ncols + ncols]):  # noqa: E203
                so = sorted({v for v in df[n].values if str(v) != "nan"})
                axs = sns.countplot(x=df[n].dropna(), ax=ax[idx], order=so)
                if idx != 0:
                    axs.set(ylabel="")


def show_target_vs_categorical(df: pd.DataFrame, target: list[str], figsize: tuple = (17, 4), ncols: int = 5) -> None:
    """
    Display bar plots of target vs categorical variables. Target values must be numerical for bar plots.
    Args:
        df (pd.DataFrame): Input dataframe
        target (list[str]): List of target variables of the input dataframe
        figsize (tuple, optional): Figure size. Defaults to (17, 4).
        ncols (int, optional): Max number of subplots in a row of plots. Defaults to 5.
    """

    numerical = list(df.select_dtypes(include=[np.number]))
    categorical_f = [col for col in df if col not in numerical and col not in target]

    if ncols <= 1:
        ncols = 5
        print(f"Number of columns changed to {ncols}")

    if not categorical_f:
        print("There are no categorical variables")
        return

    copy_df = df.copy()
    for t in target:
        copy_df = copy_df[pd.notnull(copy_df[t])]
        if t not in numerical:
            copy_df[t] = copy_df[t].astype(np.float16)

    nrows = math.ceil(len(categorical_f) / ncols)

    for t in target:  # in case of several targets several plots will be shown

        for row in range(nrows):

            if row == nrows - 1 and len(categorical_f) % ncols == 1:  # case 1 only plot in last row
                plt.subplots(ncols=1, figsize=figsize)
                # so = sorted({v for v in copy_df[f].values if str(v) != 'nan'})
                axs = sns.barplot(data=copy_df, x=categorical_f, y=t)

            else:

                if row == nrows - 1 and len(categorical_f) % ncols != 0:
                    ncols = len(categorical_f) % ncols  # adjust size of last row

                _, ax = plt.subplots(ncols=ncols, sharey=True, figsize=figsize)

                for idx, f in enumerate(categorical_f[row * ncols : row * ncols + ncols]):  # noqa: E203

                    so = sorted({v for v in copy_df[f].values if str(v) != "nan"})

                    axs = sns.barplot(data=copy_df, x=f, y=t, ax=ax[idx], order=so)

                    # first y-axis label only
                    if idx != 0:
                        axs.set(ylabel="")


def correlation(df: pd.DataFrame, target: list[str], threshold: float = 0, figsize: tuple = None) -> list[str]:
    """
    Plot the Pearson correlation coefficient between target and numerical features. Return a list with low-correlated
    features.
    Args:
        df (pd.DataFrame): Input dataframe
        target (list[str]): List of target variables of the input dataframe
        threshold (float, optional): Correlation coefficient threshold. Defaults to 0.
        figsize (tuple, optional): Figure size. Defaults to None.
    Returns:
        list[str]: List of low-correlated features (below threshold)
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

    if threshold > 0:
        plt.axvline(
            x=-threshold,
            color="k",
            linestyle="--",
        )
        plt.axvline(
            x=threshold,
            color="k",
            linestyle="--",
        )
    plt.xlabel("Pearson correlation coefficient")
    plt.ylabel("feature")

    return corr.loc[abs(corr[target[0]]) < abs(threshold)].index.tolist()


# DATA PROCESSING FOR ML & DL ---------------------------------------------


def scale(data: pd.DataFrame, scale_param: dict = None, method: str = "std") -> tuple[pd.DataFrame, dict]:
    """
    Scale/Standardize the numerical variables of the input dataset using a given dictionary of mean and standard
    deviation for each numerical variable or a default method.
    Args:
        data (pd.DataFrame): Input dataframe to standardize
        scale_param (dict, optional): Dictionary with parameters for scaling. Structure: {'var_name': [mean, std]}.
        Defaults to None.
        method (str, optional): Method for scaling. Defaults to "std" (normal distribution), "minmax" or "maxabs".
    Returns:
        pd.DataFrame: Dataframe with numerical values standardized
        dict: Dictionary with parameters for scaling. Structure: {'var_name': [mean, std]}.
    # TODO: Replace scalers with those from sklearn.preprocessing
    """

    assert method == "std" or method == "minmax" or method == "maxabs"

    data = data.copy()

    num = list(data.select_dtypes(include=[np.number]))
    if not scale_param:
        create_scale = True
        scale_param = {}
    else:
        create_scale = False

    for f in num:
        if method == "std":
            if create_scale:
                mean, std = data[f].mean(), data[f].std()
                data[f] = (data[f].values - mean) / std
                scale_param[f] = [mean, std]
            else:
                data.loc[:, f] = (data[f] - scale_param[f][0]) / scale_param[f][1]

        elif method == "minmax":

            if create_scale:
                min_value, max_value = data[f].min(), data[f].max()
                data[f] = (data[f].values - min_value) / (max_value - min_value)
                scale_param[f] = [min_value, max_value]
            else:
                min_value, max_value = scale_param[f][0], scale_param[f][1]
                data.loc[:, f] = (data[f].values - min_value) / (max_value - min_value)

        elif method == "maxabs":

            if create_scale:
                min_value, max_value = data[f].min(), data[f].max()
                data[f] = 2 * (data[f].values - min_value) / (max_value - min_value) - 1
                scale_param[f] = [min_value, max_value]
            else:
                min_value, max_value = scale_param[f][0], scale_param[f][1]
                data.loc[:, f] = 0.5 * (data[f].values * (max_value - min_value) + max_value + min_value)

    return data, scale_param


def replace_by_dummies(
    data: pd.DataFrame, target: list[str], dummies: list[str] = None, drop_first: bool = False
) -> tuple[pd.DataFrame, list[str]]:
    """
    Replace categorical features by dummy features (target variables excludes).
    If no dummy list is used, a new one is generated.
    Args:
        data (pd.DataFrame): Input dataframe
        target (list[str]): List of target variables of the input dataframe
        dummies (list[str], optional): List of dummy features. Defaults to None.
        drop_first (bool, optional): Drop first dummy feature. Defaults to False.
    Returns:
        pd.DataFrame: Dataframe with dummy features
        list[str]: List of dummy features of the new dataframe
    TODO: Replace by scikit-learn Pipelines
    """

    data = data.copy()

    if not dummies:
        create_dummies = True
    else:
        create_dummies = False

    found_dummies = []

    categorical = list(data.select_dtypes(include=["category"]))
    categorical_f = [col for col in categorical if col not in target]

    for f in categorical_f:
        dummy = pd.get_dummies(data[f], prefix=f, drop_first=drop_first)
        data = pd.concat([data, dummy], axis=1)
        data = data.drop(f, axis=1)

        found_dummies.extend(dummy)

    if not create_dummies:

        # remove new dummies not in given dummies
        new = set(found_dummies) - set(dummies)
        for n in new:
            data = data.drop(n, axis=1)

        # fill missing dummies with empty values (0)
        missing = set(dummies) - set(found_dummies)
        for m in missing:
            data[m] = 0

    else:
        dummies = found_dummies

    # set new columns to category
    for dummy in dummies:
        data[dummy] = data[dummy].astype("category")

    return data, dummies


def get_class_weight(input_array: np.Array | pd.Series) -> dict:
    """Return a dictionary of weights of the input array. Used for unbalanced data
    Args:
        input_array (np.Array|pd.Series): Input array
    Returns:
        dict: Dictionary of weights of the input array structure: {idx: weight} (idx= 0,1,2...)
    """

    input_array = np.ravel(input_array)
    weight = class_weight.compute_class_weight("balanced", classes=np.unique(input_array), y=input_array)
    weight = dict(enumerate(weight))
    # print(weight)
    return weight


def one_hot_output(
    input_array: np.Array | pd.Series, input_array_2: np.Array | pd.Series = None
) -> np.Array | tuple(np.Array, np.Array):
    """
    Return the one-hot-encoded output of the input array. If a second input_array is provided, both encoded output
    are returned (usual for y_train, y_test)
    Args:
        input_array (np.Array|pd.Series): Input array
        input_array_2 (np.Array|pd.Series, optional): Second input array. Defaults to None.
    Returns:
        np.Array: One-hot-encoded output of the input array
        (optional)  np.Array: One-hot-encoded output of the second input array if provided
    """
    num_classes = len(np.unique(input_array))
    encoded_array = keras.utils.to_categorical(input_array, num_classes)
    if input_array_2.any():
        encoded_array_2 = keras.utils.to_categorical(input_array_2, num_classes)
        return encoded_array, encoded_array_2
    else:
        return encoded_array


def data_split_for_ml(
    data: pd.DataFrame, target: list[str], stratify: bool = False, test_size: float = 0.2, random_state: int = 9
) -> tuple(np.Array, np.Array, np.Array, np.Array):
    """
    Separate the data intro training and test set. Also split them into features and target. Stratified split will use
    class labels when 'stratify=True'(classification).
    Args:
        data (pd.DataFrame): Input dataframe
        target (list[str]): List of target variables of the input dataframe
        stratify (bool, optional): Use class labels when 'stratify=True'(classification). Defaults to False.
        test_size (float, optional): Test size. Defaults to 0.2.
        random_state (int, optional): Random state. Defaults to 9.
    Returns:
        np.Array: x_train - train features
        np.Array: y_train - train target
        np.Array: x_test - test features
        np.Array: y_test - test target
    """

    label_data = data[target] if stratify else None

    train, test = train_test_split(data, test_size=test_size, random_state=random_state, stratify=label_data)

    # Separate the data into features and targets (x=features, y=targets)
    x_train, y_train = train.drop(target, axis=1).values, train[target].values
    x_test, y_test = test.drop(target, axis=1).values, test[target].values

    return x_train, y_train, x_test, y_test


def data_split_for_ml_with_val(
    data: pd.DataFrame,
    target: list[str],
    stratify: bool = False,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 9,
) -> tuple(np.Array, np.Array, np.Array, np.Array, np.Array, np.Array):
    """
    Separate the data intro training and test set. Also split them into features and target. Stratified split will use
    class labels when 'stratify=True'(classification).
    Args:
        data (pd.DataFrame): Input dataframe
        target (list[str]): List of target variables of the input dataframe
        stratify (bool, optional): Use class labels when 'stratify=True'(classification). Defaults to False.
        test_size (float, optional): Test size. Defaults to 0.2.
        random_state (int, optional): Random state. Defaults to 9.
    Returns:
        np.Array: x_train - train features
        np.Array: y_train - train target
        np.Array: x_val - validation features
        np.Array: y_val - validation target
        np.Array: x_test - test features
        np.Array: y_test - test target
    """

    label_data = data[target] if stratify else None

    train, test = train_test_split(data, test_size=test_size, random_state=random_state, stratify=label_data)
    train, val = train_test_split(train, test_size=val_size, random_state=random_state, stratify=label_data)

    # Separate the data into features and target (x=features, y=target)
    x_train, y_train = train.drop(target, axis=1).values, train[target].values
    x_val, y_val = val.drop(target, axis=1).values, val[target].values
    x_test, y_test = test.drop(target, axis=1).values, test[target].values

    print(f"train size \t X:{x_train.shape} \t Y:{y_train.shape}")
    print(f"validation size\t X:{x_val.shape} \t Y:{y_val.shape}")
    print(f"test size  \t X:{x_test.shape} \t Y:{y_test.shape} ")

    return x_train, y_train, x_val, y_val, x_test, y_test


# ML Metrics


def binary_classification_scores(
    y_test: np.Array | pd.Series,
    y_pred: np.Array | pd.Series,
    return_dataframe: bool = False,
    index: str = " ",
    show: bool = True,
) -> tuple[float, float, float, float, float, float] | pd.DataFrame:
    """Get ML classification metrics from test & predicted arrays: log_loss, acc, precision, recall, roc_auc, & F1 score
    Args:
        y_test (np.Array|pd.Series): Test target
        y_pred (np.Array|pd.Series): Predicted target
        return_dataframe (bool, optional): Return metrics in a dataframe. Defaults to False.
        index (str, optional): Index of the dataframe for visualization. Defaults to " "
        show (bool, optional): Show the dataframe. Defaults to True.
    Returns:
        tuple[float, float, float, float, float, float]: log_loss, acc, precision, recall, roc_auc, & F1 score
        or
        pd.DataFrame: Above metrics in Dataframe format (return_dataframe=True)
        # TODO extend to multiclass classification
    """

    rec, roc, f1 = 0, 0, 0

    y_pred_b = (y_pred > 0.5).astype(int)

    warnings.filterwarnings("ignore", message="divide by zero encountered in log")
    warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
    loss = log_loss(y_test, y_pred)

    acc = accuracy_score(y_test, y_pred_b)

    warnings.filterwarnings(
        "ignore",
        message="Precision is ill-defined and being set to 0.0 due to no predicted samples",
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
        print("\nConfusion matrix: \n", confusion_matrix(y_test, y_pred_b))

    if return_dataframe:
        col = ["Loss", "Accuracy", "Precision", "Recall", "ROC-AUC", "F1-score"]
        scores = pd.DataFrame([[loss, acc, pre, rec, roc, f1]], columns=col, index=[index]).round(2)
        return scores

    return loss, acc, pre, rec, roc, f1


def regression_scores(
    y_test: np.Array | pd.Series,
    y_pred: np.Array | pd.Series,
    return_dataframe: bool = False,
    index: str = " ",
    show: bool = False,
) -> tuple[float, float] | pd.DataFrame:
    """Get ML regression metrics from test & predicted arrays: loss & R2 Score
    Args:
        y_test (np.Array|pd.Series): Test target
        y_pred (np.Array|pd.Series): Predicted target
        return_dataframe (bool, optional): Return metrics in a dataframe. Defaults to False.
        index (str, optional): Index of the dataframe for visualization. Defaults to " "
        show (bool, optional): Show the dataframe. Defaults to True.
    Returns:
        tuple[float, float]: loss & R2 Score
        or
        pd.DataFrame: Above metrics in Dataframe format (return_dataframe=True)
    """
    r2 = r2_score(y_test, y_pred)
    loss = mean_squared_error(y_test, y_pred)

    if show:
        print("Scores:\n" + "-" * 11)
        print(f"Loss (mse): \t{loss:.4f}")
        print(f"R2 Score: \t{r2:.2f}")

    if return_dataframe:
        col = ["Loss", "R2 Score"]
        scores = pd.DataFrame([[loss, r2]], columns=col, index=[index]).round(2)
        return scores

    return loss, r2


# ML/DL MODELING TODO: Finish docstrings


def dummy_clf(
    x_train: np.Array | pd.Series,
    y_train: np.Array | pd.Series,
    x_test: np.Array | pd.Series,
    y_test: np.Array | pd.Series,
) -> tuple[float, float, float, float, float, float] | pd.DataFrame:
    """
    Build a dummy classifier, print the confusion matrix and return the binary classification scores
    Args:
        x_train (np.Array|pd.Series): Training features
        y_train (np.Array|pd.Series): Training target
        x_test (np.Array|pd.Series): Test features
        y_test (np.Array|pd.Series): Test target
    Returns:
        tuple[float, float, float, float, float, float]: log_loss, acc, precision, recall, roc_auc, & F1 score
        or
        pd.DataFrame: Above metrics in Dataframe format (return_dataframe=True)
    """

    clf = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    y_pred = clf.predict(x_test)  # .reshape([-1, 1])

    return binary_classification_scores(y_test, y_pred, return_dataframe=True, index="Dummy")


def build_nn_binary_clf(
    input_size,
    hidden_layers=1,
    dropout=0,
    input_nodes=None,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    # kernel_regularizer=None,
    # bias_regularizer=None,
):
    """Build a DNN for binary classification with sigmoid activation function"""

    if not input_nodes:
        input_nodes = input_size

        # weights = keras.initializers.RandomNormal(stddev=0.00001)

    model = Sequential()

    # input + first hidden layer
    model.add(
        Dense(
            input_nodes,
            input_dim=input_size,
            activation="relu",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )
    )
    model.add(Dropout(dropout))

    # additional hidden layers
    for i in range(1, hidden_layers):
        model.add(
            Dense(
                input_nodes // i + 1,
                activation="relu",
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
            )
        )
        model.add(Dropout(dropout))

    # output layer
    model.add(
        Dense(
            1,
            activation="sigmoid",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )
    )


def build_nn_clf(
    input_size,
    output_size,
    hidden_layers=1,
    dropout=0,
    input_nodes=None,
    summary=False,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    # kernel_regularizer=None,
    # bias_regularizer=None,
):
    """Build an universal DNN for classification"""

    if not input_nodes:
        input_nodes = input_size
        # weights = keras.initializers.RandomNormal(stddev=0.00001)

    model = Sequential()

    # input + first hidden layer
    model.add(
        Dense(
            input_nodes,
            input_dim=input_size,
            activation="relu",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )
    )
    model.add(Dropout(dropout))

    # additional hidden layers
    for i in range(1, hidden_layers):
        model.add(
            Dense(
                input_nodes // i + 1,
                activation="relu",
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
            )
        )
        model.add(Dropout(dropout))

    # output layer
    model.add(
        Dense(
            output_size,
            activation="softmax",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )
    )

    opt = keras.optimizers.Adam()
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    if summary:
        model.summary()

    return model


def build_nn_reg(
    input_size,
    output_size,
    hidden_layers=1,
    dropout=0,
    input_nodes=None,
    summary=False,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    # kernel_regularizer=None,
    # bias_regularizer=None,
    optimizer="rmsprop",
):
    """Build an universal DNN for regression"""

    if not input_nodes:
        input_nodes = input_size

        # weights = keras.initializers.RandomNormal(stddev=0.00001)

    model = Sequential()

    # input + first hidden layer
    model.add(
        Dense(
            input_nodes,
            input_dim=input_size,
            activation="relu",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )
    )
    model.add(Dropout(dropout))

    # additional hidden layers
    for i in range(1, hidden_layers):
        model.add(
            Dense(
                input_nodes // i + 1,
                activation="relu",
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
            )
        )
        model.add(Dropout(dropout))

    # output layer
    model.add(
        Dense(
            output_size,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )
    )

    model.compile(loss="mean_squared_error", optimizer=optimizer)

    if summary:
        model.summary()

    return model


def train_nn(
    model,
    x_train,
    y_train,
    cw=None,
    epochs=100,
    batch_size=128,
    verbose=0,
    callbacks=None,
    validation_split=0.2,
    validation_data=None,
    path=False,
    show=True,
):
    """
    Train a neural network model. If no validation_data is provided, a split for validation
    will be used
    """

    if show:
        print("Training ....")

    # callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, verbose=0)]
    t0 = time()

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        class_weight=cw,
        validation_split=validation_split,
        validation_data=validation_data,
        callbacks=callbacks,
    )

    if show:
        print(f"time: \t {time() - t0:.1f} s")
        show_training(history)

    if path:
        model.save(path)
        print("\nModel saved at", path)

    return history


def show_training(history):
    """
    Print the final loss and plot its evolution in the training process.
    The same applies to 'validation loss', 'accuracy', and 'validation accuracy' if available
    :param history: Keras history object (model.fit return)
    :return:
    """
    hist = history.history

    if "loss" not in hist:
        print("Error: 'loss' values not found in the history")
        return

    # plot training
    plt.figure(figsize=(14, 4))
    plt.subplot(121)
    plt.plot(hist["loss"], label="Training")
    if "val_loss" in hist:
        plt.plot(hist["val_loss"], label="Validation")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()

    if "accuracy" in hist:
        plt.subplot(122)
        plt.plot(hist["accuracy"], label="Training")
        if "val_accuracy" in hist:
            plt.plot(hist["val_accuracy"], label="Validation")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend()

    plt.show()

    # show final results
    print(f"\nTraining loss:  \t{hist['loss'][-1]:.4f}")
    if "val_loss" in hist:
        print(f"Validation loss: \t {hist['val_loss'][-1]:.4f}")
    if "accuracy" in hist:
        print(f"\nTraining accuracy: \t{hist['accuracy'][-1]:.3f}")
    if "val_accuracy" in hist:
        print(f"Validation accuracy:\t{hist['val_accuracy'][-1]:.3f}")


def ml_classification(x_train, y_train, x_test, y_test, cross_validation=False, show=False):
    """
    Build, train, and test the data set with classical machine learning classification models.
    If cross_validation=True an additional training with cross validation will be performed.
    """

    classifiers = (
        GaussianNB(),
        RandomForestClassifier(100),
        ExtraTreesClassifier(100),
        LGBMClassifier(n_jobs=-1, n_estimators=30, max_depth=17),
    )

    names = ["Naive Bayes", "Random Forest", "Extremely Randomized Trees", "LGBM"]

    col = ["Time (s)", "Loss", "Accuracy", "Precision", "Recall", "ROC-AUC", "F1-score"]
    results = pd.DataFrame(columns=col)

    for idx, clf in enumerate(classifiers):

        # clf_cv = clone(clf)

        name = names[idx]
        print(name)

        t0 = time()
        # Fitting the model without cross validation

        warnings.filterwarnings("ignore", message="overflow encountered in reduce")

        clf.fit(x_train, y_train)
        train_time = time() - t0
        y_pred = clf.predict_proba(x_test)

        loss, acc, pre, rec, roc, f1 = binary_classification_scores(y_test, y_pred[:, 1], show=show)

        if cross_validation:
            warnings.warn("Cross-validation removed")  # TODO: Update with best practices 2022 (ethics & green code)

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

        results = pd.concat(
            [results, pd.DataFrame([[train_time, loss, acc, pre, rec, roc, f1]], columns=col, index=[name])]
        )

    return results.sort_values("Accuracy", ascending=False).round(2)


def ml_regression(x_train, y_train, x_test, y_test, cross_validation=False, show=False):
    """
    Build, train, and test the data set with classical machine learning regression models.
    If cross_validation=True an additional training with cross validation will be performed.
    """

    regressors = [
        LinearRegression(),
        KNeighborsRegressor(n_neighbors=10),
        # AdaBoostRegressor(),
        RandomForestRegressor(max_depth=17),
    ]

    names = ["Linear", "KNeighbors", "Random Forest"]

    if y_train.shape[1] == 1:
        regressors.append(LGBMRegressor(n_jobs=-1, n_estimators=30, max_depth=17))
        names.append("LGBM")

    col = ["Time (s)", "Test loss", "Test R2 score"]
    results = pd.DataFrame(columns=col)

    for idx, clf in enumerate(regressors):

        name = names[idx]
        # clf_cv = clone(clf)

        print(name)

        t0 = time()
        # Fitting the model without cross validation
        clf.fit(x_train, y_train.ravel())
        train_time = np.around(time() - t0, 1)
        y_pred = clf.predict(x_test)

        loss, r2 = regression_scores(y_test, y_pred, show=show)

        if cross_validation:
            warnings.warn("Cross-validation removed")  # TODO: Update with best practices 2022 (ethics & green code)

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

        results = pd.concat([results, pd.DataFrame([[train_time, loss, r2]], columns=col, index=[name])])

        if show:
            print("-" * 20)
            print(f"Training Time:  \t {train_time:.1f} s")
            print(f"Test loss:  \t\t {loss:.4f}")
            print(f"Test R2-score:  \t {r2:.3f}\n")

    return results.sort_values("Test loss").round(2)


def feature_importance(features, model, top=10, plot=True):
    """Return a dataframe with the most relevant features from a trained tree-based model"""

    n = len(features)
    if n < top:
        top = n

    # print("\n Top contributing features:\n", "-" * 26)
    importance = pd.DataFrame(data={"Importance": model.feature_importances_}, index=features)
    importance = importance.sort_values("Importance", ascending=False).round(2).head(top)

    if plot:
        figsize = (8, top // 2 + 1)
        importance.plot.barh(figsize=figsize)
        plt.gca().invert_yaxis()

    return importance
