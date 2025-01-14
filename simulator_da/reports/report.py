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
from collections import namedtuple

sns.set(rc={'figure.figsize': (11, 8)}, style='whitegrid')

DAILY_CHAT_ID=os.environ.get("DAILY_CHAT_ID")
REPORT_BOT_TOKEN=os.environ.get("REPORT_BOT_TOKEN")

bot = telegram.Bot(REPORT_BOT_TOKEN)


def report(chat_identificator):
    """
    Compiles report consisting of DAU, Likes, Views and CTR values from
    yesterday and their graphs for last week, then sends it to the chat with
    chat_identificator id via @kc_iorlov_report_bot.
    """

    def df_plot_send_calc(df, metric):
        """
        Plots lineplot with title plot_title and Y axis
        formatted according to y_format_string
        Calculates yesterday value of key metric and
        returns its string representation with title

        Parameters
        ----------
        df : DataFrame
            dataframe with needed metric and timestamp

        metric : Metric
            name : str
                Metric name as in query and resulting dataframe
            title : str
                Title to use in plot and result string
            y_format_string  : str
                Format string to use for Y values in plot and result string
        """
        df_plot = sns.lineplot(x=df.iloc[:, 0].map(lambda x: x.strftime('%B %d ')), y=df.iloc[:, 1])
        df_plot.set_xlabel('')
        df_plot.set_ylabel('')
        df_plot.set_title(metric.title)

        ylabels = [metric.y_axis_format_string.format(y) for y in df_plot.get_yticks()]
        df_plot.set_yticklabels(ylabels)

        plot_obj = io.BytesIO()
        plt.savefig(plot_obj)
        plot_obj.name = f"{metric.title}.png"
        plot_obj.seek(0)
        plt.close()
        bot.send_photo(chat_id=chat_identificator, photo=plot_obj)

        yesterday = datetime.now() - timedelta(days=1)
        yesterday_pd = pd.Timestamp(yesterday.strftime('%Y-%m-%d'))
        yesterday_query = 'ts == @yesterday_pd'

        return f"""{metric.title}: {metric.y_axis_format_string.format(df.query(yesterday_query).iloc[0, 1])}\n"""


    query = """SELECT toStartOfDay(toDateTime(time)) AS ts,
               count(DISTINCT user_id) AS distinct_users,
               countIf(user_id, action = 'view') AS views,
               countIf(user_id, action = 'like') AS likes,
               likes/views as CTR
               FROM simulator_20220320.feed_actions
               WHERE time >= toStartOfDay(now() - (7 * 24 * 60 * 60))
                 AND time < toStartOfDay(now())
               GROUP BY ts
               order by ts
               """

    thousand_commas_format = '{:,.0f}'
    three_after_point_format = '{:.3f}'

    Metric = namedtuple('Metric', ['title', 'name', 'y_axis_format_string'])

    DAU = Metric(title='DAU',
                 name='distinct_users',
                 y_axis_format_string=thousand_commas_format)
    Likes = Metric(title='Likes',
                   name='likes',
                   y_axis_format_string=thousand_commas_format)
    Views = Metric(title='Views',
                   name='views',
                   y_axis_format_string=thousand_commas_format)
    CTR = Metric(title='CTR',
                 name='CTR',
                 y_axis_format_string=three_after_point_format)


    metrics = (DAU, Likes, Views, CTR)

    query_df = Getch(query).df

    key_metrics = f"""Key metrics' values for yesterday ({(datetime.now() - timedelta(days=1)).strftime("%d %B %Y")}):\n"""

    for metric in metrics:
        key_metrics += df_plot_send_calc(query_df[['ts', metric.name]], metric)

    bot.send_message(chat_id=chat_identificator, text=key_metrics)


try:
    report(DAILY_CHAT_ID)
except Exception as e:
    print(e)
