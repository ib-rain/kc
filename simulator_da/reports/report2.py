#!/usr/bin/python
# -*- coding: utf-8 -*-
import telegram
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from read_db.CH import Getch
import os
import pandas as pd
from datetime import datetime, timedelta
import time
from collections import namedtuple

sns.set(rc={'figure.figsize': (15, 10)}, style='whitegrid', font_scale=0.8)

DAILY_CHAT_ID = os.environ.get("DAILY_CHAT_ID")
REPORT_BOT_TOKEN = os.environ.get("REPORT_BOT_TOKEN")

bot = telegram.Bot(REPORT_BOT_TOKEN)

LAST_DAY_STR = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
LAST_WEEK_STR = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')


def q25(x):
    return x.quantile(0.25)


def q75(x):
    return x.quantile(0.75)


def cv(x):
    if x.mean() == 0:
        return None
    return x.std() / x.mean()


def report(chat_identificator):
    """
    Compiles report consisting of 'Active Users - Feed', 'Views', 'Likes',
    'CTR', 'Unique Viewed Posts', 'Active Users - Messenger', 'Sent Messages',
    'Active Users - Both' values from yesterday and for the last week,
    calculates statistics for it and plots graphs, then sends it to the chat
    with chat_identificator id via @kc_iorlov_report_bot.
    """

    Metric = namedtuple('Metric', ['title', 'name', 'y_axis_format_string'])

    # Amount of seconds in the timestep for last_day_query
    last_day_atom_s = 60 * 60

    thousands_commas_format = '{:,.0f}'
    three_digits_after_point_format = '{:.3f}'

    def metrics_plot_send_calc(metrics, df_day, df_week):
        """
        Getch_s data frames according to metric.last_week_query and metric.last_day_query.
        Plots lineplots with title metric.name and Y axis formatted according to metric.y_axis_format_string.
        Calculates (mean, std, cv, min, q25, median, q75, max) of key metric last week and last day values,
        returns its pd.Series representations as tuple.

        Parameters
        ----------
        metrics : tuple(Metric)

        df : DataFrame
        """
        n_of_metrics = len(metrics)

        colours = ('green', 'blue', 'red', 'black')

        def plot_df_metrics(df, time_colname, time_strf, date_str):
            (figure, axes) = plt.subplots(n_of_metrics, 1, figsize=(20, 15), sharex=True)
            figure.suptitle(date_str, size=20)
            for i in range(n_of_metrics):
                df_plot = sns.lineplot(x=df[time_colname].map(lambda x: x.strftime(time_strf)),
                                       y=df[metrics[i].name], ax=axes[i], color=colours[i])
                df_plot.set_xlabel('')
                df_plot.set_ylabel('')
                df_plot.set_title(metrics[i].title, size=13)
                ylabels = [metrics[i].y_axis_format_string.format(y) for y in df_plot.get_yticks()]
                df_plot.set_yticklabels(ylabels)

            plot_obj = io.BytesIO()
            plt.savefig(plot_obj)
            plot_obj.name = metrics[i].title + '.png'
            plot_obj.seek(0)

            plt.close()
            bot.send_photo(chat_id=chat_identificator, photo=plot_obj)


        plot_df_metrics(df_day, 'ts', "%H:%M", f'{LAST_DAY_STR}')
        plot_df_metrics(df_week, "ts_day", "%B %d", f'Week {LAST_WEEK_STR}_{LAST_DAY_STR}')

        stat_day = pd.DataFrame()
        stat_week = pd.DataFrame()

        stat_tuple = ('mean', 'std', cv, 'min', q25, 'median', q75, 'max')

        for metric in metrics:
            stat_day = stat_day.append(df_day[metric.name].agg(stat_tuple))
            stat_week = stat_week.append(df_week[metric.name].agg(stat_tuple))

        return (stat_day, stat_week)


    def send_df_as_csv(df, name):
        file_object = io.StringIO()
        df.to_csv(file_object)
        file_object.name = name + '.csv'
        file_object.seek(0)
        bot.sendDocument(chat_id=chat_identificator, document=file_object)


    feed_query_day="""SELECT toDateTime(intDiv(toUInt32(toDateTime(time)), {atom_s})*{atom_s}) AS ts,
              count(DISTINCT user_id) AS active_users_feed,
              countIf(user_id, action = 'view') AS views,
              countIf(user_id, action = 'like') AS likes,
              likes/views as CTR,
              countIf(distinct post_id, action = 'view') AS viewed_posts
              FROM simulator_20220320.feed_actions
              WHERE time >= toStartOfDay(now() - (1 * 24 * 60 * 60))
                AND time < toStartOfDay(now())
              GROUP BY ts
              order by ts
              """.format(atom_s=last_day_atom_s)

    feed_query_week="""SELECT toStartOfDay(toDateTime(time)) AS ts_day,
              count(DISTINCT user_id) AS active_users_feed,
              countIf(user_id, action = 'view') AS views,
              countIf(user_id, action = 'like') AS likes,
              likes/views as CTR,
              countIf(distinct post_id, action = 'view') AS viewed_posts
              FROM simulator_20220320.feed_actions
              WHERE time >= toStartOfDay(now() - (7 * 24 * 60 * 60))
                AND time < toStartOfDay(now())
              GROUP BY ts_day
              order by ts_day
              """

    AU_feed = Metric(title='Active Users - Feed',
                 name='active_users_feed',
                 y_axis_format_string=thousands_commas_format)

    Views = Metric(title='Views',
                   name='views',
                   y_axis_format_string=thousands_commas_format)

    Likes = Metric(title='Likes',
                   name='likes',
                   y_axis_format_string=thousands_commas_format)

    CTR = Metric(title='CTR',
                 name='CTR',
                 y_axis_format_string=three_digits_after_point_format)

    UVP = Metric(title='Unique Viewed Posts',
                 name='viewed_posts',
                 y_axis_format_string=thousands_commas_format)


    msg_query_day="""SELECT toDateTime(intDiv(toUInt32(toDateTime(time)), {atom_s})*{atom_s}) AS ts,
             count(distinct user_id) AS active_users_msg,
             count(distinct user_id,reciever_id,time) AS sent_messages
             FROM simulator_20220320.message_actions
             WHERE time >= toStartOfDay(now() - (1 * 24 * 60 * 60))
               AND time < toStartOfDay(now())
             GROUP BY ts
             order by ts
             """.format(atom_s=last_day_atom_s)

    msg_query_week="""SELECT toStartOfDay(toDateTime(time)) AS ts_day,
             count(distinct user_id) AS active_users_msg,
             count(distinct user_id,reciever_id,time) AS sent_messages
             FROM simulator_20220320.message_actions
             WHERE time >= toStartOfDay(now() - (7 * 24 * 60 * 60))
               AND time < toStartOfDay(now())
             GROUP BY ts_day
             order by ts_day
             """

    AU_msg = Metric(title='Active Users - Messenger',
                    name='active_users_msg',
                    y_axis_format_string=thousands_commas_format)

    SM = Metric(title='Sent Messages',
                name='sent_messages',
                y_axis_format_string=thousands_commas_format)


    both_query_day="""SELECT toDateTime(intDiv(toUInt32(toDateTime(time)), {atom_s})*{atom_s}) AS ts,
             count(DISTINCT f.user_id) AS active_users_both
             from simulator_20220320.feed_actions as f
             inner join simulator_20220320.message_actions as m using(user_id)
             WHERE time >= toStartOfDay(now() - (1 * 24 * 60 * 60))
               AND time < toStartOfDay(now())
             GROUP BY ts
             order by ts
             """.format(atom_s=last_day_atom_s)

    both_query_week="""SELECT toStartOfDay(toDateTime(time)) AS ts_day,
             count(DISTINCT f.user_id) AS active_users_both
             from simulator_20220320.feed_actions as f
             inner join simulator_20220320.message_actions as m using(user_id)
             WHERE time >= toStartOfDay(now() - (7 * 24 * 60 * 60))
               AND time < toStartOfDay(now())
             GROUP BY ts_day
             order by ts_day
             """

    AU_both = Metric(title='Active Users - Both',
                     name='active_users_both',
                     y_axis_format_string=thousands_commas_format)

    metrics_stats_last_week = pd.DataFrame()
    metrics_stats_last_day = pd.DataFrame()

    mega_df_day = Getch(feed_query_day).df.merge(Getch(both_query_day).df, how='inner', on='ts').merge(Getch(msg_query_day).df, how='inner', on='ts')
    mega_df_week = Getch(feed_query_week).df.merge(Getch(both_query_week).df, how='inner', on='ts_day').merge(Getch(msg_query_week).df, how='inner', on='ts_day')

    for metric_tuple in ((AU_feed, Views, Likes, CTR),
                         (AU_both, UVP, AU_msg, SM)):
        metric_stats = metrics_plot_send_calc(metric_tuple, mega_df_day, mega_df_week)

        metrics_stats_last_day = metrics_stats_last_day.append(metric_stats[0])
        metrics_stats_last_week = metrics_stats_last_week.append(metric_stats[1])

    send_df_as_csv(metrics_stats_last_day, f'metrics_stats_{LAST_DAY_STR}')
    send_df_as_csv(metrics_stats_last_week, f'metrics_stats_week_{LAST_WEEK_STR}_{LAST_DAY_STR}')


time.sleep(13)
try:
    report(DAILY_CHAT_ID)
except Exception as e:
    print(e)
