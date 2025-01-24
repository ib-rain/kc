import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats
from datetime import datetime, timedelta

#import os


"""
Config, constants, functions that are shared by all solution notebooks.
"""

### Chart configs.
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


### DB file configs.
URL_BASE = 'https://raw.githubusercontent.com/ab-courses/simulator-ab-datasets/main/2022-04-01/{}'

# def read_database(file_name):
    #return pd.read_csv(os.path.join(URL_BASE, file_name))


def read_from_database(file_name, parse_dates_list=[]):
    return pd.read_csv(URL_BASE.format(file_name), parse_dates=parse_dates_list)


### Lesson 1.
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


### Lesson 3.
def get_minimal_determinable_effect(std, sample_size, alpha=0.05, beta=0.2):
    t_alpha = norm.ppf(1 - alpha / 2, loc=0, scale=1)
    t_beta = norm.ppf(1 - beta, loc=0, scale=1)
    
    disp_sum_sqrt = (2 * (std ** 2)) ** 0.5
    mde = (t_alpha + t_beta) * disp_sum_sqrt / np.sqrt(sample_size)

    return mde


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


# def check_ttest(a, b, alpha=0.05):
#     """Тест Стьюдента. Возвращает 1, если отличия значимы."""
#     _, pvalue = ttest_ind(a, b)
#     return int(pvalue < alpha)


def check_test(test, a, b, alpha=0.05):
    """Возвращает 1, если отличия значимы."""
    return int(test(a, b).pvalue < alpha)


### Lesson 4.
def plot_pvalue_ecdf(pvalues, title=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    if title:
        plt.suptitle(title)

    sns.histplot(pvalues, ax=ax1, bins=20, stat='density')
    ax1.plot([0,1], [1,1], 'k--')
    ax1.set(xlabel='p-value', ylabel='Density')

    sns.ecdfplot(pvalues, ax=ax2)
    ax2.plot([0,1], [0,1], 'k--')
    ax2.set(xlabel='p-value', ylabel='Probability')
    ax2.grid()
    
    
def print_kwargs(**kwargs):
    """Prints keyword agruments with their names (thus, requires named argument call)."""
    for (k, v) in kwargs.items():
        print('{} = {}'.format(k, v))


### Lesson 5.
def get_quantile_diff(values_a, values_b, quantile: float=0.999):
    return np.quantile(values_b, quantile) - np.quantile(values_a, quantile)


def get_bootstrap_point_estimates(values_a, values_b, quantile: float=0.999, N: int=1_000):
    bootstrap_point_estimates = []
    
    for _ in range(N):
        bootstrap_values_a = np.random.choice(values_a, size=len(values_a), replace=True)
        bootstrap_values_b = np.random.choice(values_b, size=len(values_b), replace=True)
        bootstrap_point_estimates.append(get_quantile_diff(bootstrap_values_a, bootstrap_values_b, quantile))
            
    return bootstrap_point_estimates


# Optimized code.
def get_bootstrap_confidence_interval(
    bootstrap_values: np.array,
    point_estimate: float=0.0,
    ci_type: str='normal',
    alpha: float=0.05
):
    """
    Bulding confidence interval.
    
    :param bootstrap_values: bootstrapped values of the metric.
    :param point_estimate: point estimate of the metric.
    :param ci_type: confidence interval type.
    :param alpha: level of significance.
    
    :returns: (left, right) - tuple containing confidence interval ends.
    """
    left, right = None, None
    
    if ci_type == 'normal':
        ppf_ = stats.norm.ppf(1.0 - alpha / 2)
        std_ = np.std(bootstrap_values)
        
        left, right = point_estimate - ppf_ * std_, point_estimate + ppf_ * std_

    elif ci_type == 'percentile':
        left, right = np.quantile(bootstrap_values, [alpha / 2, 1.0 - alpha / 2])
    
    elif ci_type == 'pivotal':
        left, right = 2 * point_estimate - np.quantile(bootstrap_values, [1.0 - alpha / 2, alpha / 2])
    
    return (left, right)


