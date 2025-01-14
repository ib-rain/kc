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

sns.set(rc={'figure.figsize': (18, 11)}, style='whitegrid', font_scale=0.75)

ANOMALY_CHAT_ID = os.environ.get("ANOMALY_CHAT_ID")
REPORT_BOT_TOKEN = os.environ.get("REPORT_BOT_TOKEN")

bot = telegram.Bot(REPORT_BOT_TOKEN)


def detect_n_mean_a_sigmas(df, metric_name, n=8, a=5):
    # DFs must not include the latest 15m because it has not been fully
    # filled yet: checking the one before last.
    # This is a tradeoff between "everything is anomally" and "app died
    # 15 minutes ago and we understand it only now".
    # In that case checking every 10 or 5 minutes might be a good idea.
    df.loc[:,'n_mean'] = df.loc[:,metric_name].shift(1).rolling(n).mean()
    df.loc[:,'n_std'] = df.loc[:,metric_name].shift(1).rolling(n).std()
    
    df.loc[:,'upper'] = df.loc[:,'n_mean'] + a * df.loc[:,'n_std']
    df.loc[:,'lower'] = df.loc[:,'n_mean'] - a * df.loc[:,'n_std']
    
    df.loc[:,'upper'] = df.loc[:,'upper'].rolling(n, center=True, min_periods=1).mean()
    df.loc[:,'lower'] = df.loc[:,'lower'].rolling(n, center=True, min_periods=1).mean()
    
    df.loc[:,'lower'] = df.loc[:,'lower'].transform(lambda x: max(x, 0))
    
    is_anomalous = False
    boundary = 0
    today_value = df.loc[:,metric_name].iloc[-1]
    
    if today_value < df.loc[:,'lower'].iloc[-1]:
        is_anomalous = True
        boundary = df.loc[:,'lower'].iloc[-1]

    
    if today_value > df.loc[:,'upper'].iloc[-1]:
        is_anomalous = True
        boundary = df.loc[:,'upper'].iloc[-1]

    return (is_anomalous, today_value, boundary, df)


def run_alerts(check_func, chat_id):
    Metric = namedtuple('Metric', ['title', 'name', 'y_axis_format_string', 'chart_url'])
    
    fifteen_min_atom_s = 60 * 15

    thousands_commas_format = '{:,.0f}'
    three_digits_after_point_format = '{:.3f}'

    time_strf = '%H:%M'
    
    feed_metrics_query="""SELECT toDateTime(intDiv(toUInt32(toDateTime(time)), {atom_s})*{atom_s}) AS ts,
                     count(DISTINCT user_id) AS active_users_feed,
                     countIf(user_id, action = 'view') AS views,
                     countIf(user_id, action = 'like') AS likes,
                     likes / views AS CTR
                     FROM simulator_20220320.feed_actions
                     WHERE ts >= toStartOfFifteenMinutes(now() - (12 * 60 * 60))
                     AND ts < toStartOfFifteenMinutes(now())
                     GROUP BY ts
                     ORDER BY ts
                     """.format(atom_s=fifteen_min_atom_s)
    
    msg_metrics_query="""SELECT toDateTime(intDiv(toUInt32(toDateTime(time)), {atom_s})*{atom_s}) AS ts,
                     count(DISTINCT user_id) AS active_users_msg,
                     count(distinct user_id, reciever_id, time) AS sent_messages
                     FROM simulator_20220320.message_actions
                     WHERE ts >= toStartOfFifteenMinutes(now() - (12 * 60 * 60))
                     AND ts < toStartOfFifteenMinutes(now())
                     GROUP BY ts
                     ORDER BY ts
                     """.format(atom_s=fifteen_min_atom_s)

    AU_feed = Metric(title='Active Users - Feed',
                     name='active_users_feed',
                     y_axis_format_string=thousands_commas_format,
                     chart_url='http://superset.lab.karpov.courses/r/742')

    Views = Metric(title='Views',
                   name='views',
                   y_axis_format_string=thousands_commas_format,
                   chart_url='http://superset.lab.karpov.courses/r/745')

    Likes = Metric(title='Likes',
                   name='likes',
                   y_axis_format_string=thousands_commas_format,
                   chart_url='http://superset.lab.karpov.courses/r/744')

    CTR = Metric(title='CTR',
                 name='CTR',
                 y_axis_format_string=three_digits_after_point_format,
                 chart_url='http://superset.lab.karpov.courses/r/743')

    AU_msg = Metric(title='Active Users - Messenger',
                    name='active_users_msg',
                    y_axis_format_string=thousands_commas_format,
                    chart_url='http://superset.lab.karpov.courses/r/746')
    
    SM = Metric(title='Sent Messages',
                name='sent_messages',
                y_axis_format_string=thousands_commas_format,
                chart_url='http://superset.lab.karpov.courses/r/747')

    metrics = (AU_feed, Views, Likes, CTR, AU_msg, SM)
    
    full_df = Getch(feed_metrics_query).df.merge(Getch(msg_metrics_query).df, how='inner', on='ts')

    for metric in metrics:
        df = full_df[['ts', metric.name]]
        (is_anomalous, metric_val, boundary, df) = check_func(df, metric.name)

        if is_anomalous:
            msg = f"""Metric "{metric.title}" at {df.ts.max().strftime("%H:%M (%Y/%m/%d)")}:\ncurrent value {metric_val:,.3f}\nboundary {boundary:,.3f}\ndifference {metric_val-boundary:,.3f} ({(metric_val/boundary-1)*100:.2f}% overhead)\nView chart at {metric.chart_url}"""
            
            df = df[df.ts >= df.ts.max() - pd.DateOffset(hours=9)]
                        
            df_plot = sns.lineplot(x=df['ts'], y=df[metric.name], label='metric')
            df_plot = sns.lineplot(x=df['ts'], y=df['upper'], label='upper')
            df_plot = sns.lineplot(x=df['ts'], y=df['lower'], label='lower')
            
            df_plot.set_xlabel('time')
            df_plot.set_ylabel('')
            df_plot.set_title(metric.title, size=13)
            
            df_plot.set(ylim=(0, None))
            ylabels = [metric.y_axis_format_string.format(y) for y in df_plot.get_yticks()]
            df_plot.set_yticklabels(ylabels)
            
            xlabels = [pd.to_datetime(x, unit='D').strftime(time_strf) for x in df_plot.get_xticks()]
            df_plot.set_xticklabels(xlabels)

            plot_obj = io.BytesIO()
            plt.savefig(plot_obj)
            plot_obj.name = f'{metric.name}.png'
            plot_obj.seek(0)
            plt.close()
            bot.send_photo(chat_id=chat_id, photo=plot_obj, caption=msg)
            

try:
    run_alerts(check_func=detect_n_mean_a_sigmas, chat_id=ANOMALY_CHAT_ID)
except Exception as e:
    print(e)
