import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.stats as stats
from datetime import datetime, timedelta

#import os


"""
Config, constants, functions that are shared by all solution notebooks.
"""

titlesize = 16
labelsize = 16
legendsize = 16
xticksize = 16
yticksize = xticksize

plt.rcParams['legend.markerscale'] = 1.5     # the relative size of legend markers vs. original
plt.rcParams['legend.handletextpad'] = 0.5
plt.rcParams['legend.labelspacing'] = 0.4    # the vertical space between the legend entries in fraction of fontsize
plt.rcParams['legend.borderpad'] = 0.5       # border whitespace in fontsize units
plt.rcParams['font.size'] = 12
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['axes.labelsize'] = labelsize
plt.rcParams['axes.titlesize'] = titlesize
plt.rcParams['figure.figsize'] = (10, 6)

plt.rc('xtick', labelsize=xticksize)
plt.rc('ytick', labelsize=yticksize)
plt.rc('legend', fontsize=legendsize)


URL_BASE = 'https://raw.githubusercontent.com/ab-courses/simulator-ab-datasets/main/2022-04-01/{}'

# def read_database(file_name):
    #return pd.read_csv(os.path.join(URL_BASE, file_name))


def read_from_database(file_name, parse_dates_list=[]):
    return pd.read_csv(URL_BASE.format(file_name), parse_dates=parse_dates_list)


def get_data_subset(df, begin_date=None, end_date=None, user_ids=None, columns=None):
    """Возвращает подмножество данных.

    :param df (pd.DataFrame): таблица с данными, обязательные столбцы: 'date', 'user_id'.
    :param begin_date (datetime.datetime | None): дата начала интервала с данными.
        Пример, df[df['date'] >= begin_date].
        Если None, то фильтровать не нужно.
    :param end_date (datetime.datetime | None): дата окончания интервала с данными.
        Пример, df[df['date'] < end_date].
        Если None, то фильтровать не нужно.
    :param user_ids (list[str] | None): список user_id, по которым нужно предоставить данные.
        Пример, df[df['user_id'].isin(user_ids)].
        Если None, то фильтровать по user_id не нужно.
    :param columns (list[str] | None): список названий столбцов, по которым нужно предоставить данные.
        Пример, df[columns].
        Если None, то фильтровать по columns не нужно.

    :return df (pd.DataFrame): датафрейм с подмножеством данных.
    """
    begin_date_, end_date_, user_ids_, columns_ = begin_date, end_date, user_ids, columns
    
    if not begin_date_:
        begin_date_ = df['date'].min()
    if not end_date_:
        end_date_ = df['date'].max() + timedelta(days=1)
    if not user_ids_:
        user_ids_ = df['user_id'].unique()
    if not columns_:
        columns_ = df.columns.to_list()
    
    return (
        df
        .loc[
            (df['date'] >= begin_date_)
            & (df['date'] < end_date_)
            & df['user_id'].isin(user_ids_),
            columns_
        ].copy()
    )


def get_sample_size_abs(epsilon, std, alpha=0.05, beta=0.2):
    """Absolute change."""
    t_alpha = stats.norm.ppf(1 - alpha / 2, loc=0, scale=1)
    t_beta = stats.norm.ppf(1 - beta, loc=0, scale=1)
    z_scores_sum_squared = (t_alpha + t_beta) ** 2
    
    sample_size = int(
        np.ceil(
            z_scores_sum_squared * (2 * std ** 2) / (epsilon ** 2)
        )
    )
    
    return sample_size


# def get_sample_size_arb(mu, std, eff=1.01, alpha=0.05, beta=0.2):
#     epsilon = (eff - 1) * mu


def get_sample_size_rel(mu, std, eff=0.01, alpha=0.05, beta=0.2):
    """Relative change."""
    return get_sample_size_abs(eff * mu, std, alpha, beta)


def check_ttest(a, b, alpha=0.05):
    """Тест Стьюдента. Возвращает 1, если отличия значимы."""
    _, pvalue = ttest_ind(a, b)
    return int(pvalue < alpha)
