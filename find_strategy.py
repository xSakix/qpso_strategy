import numpy as np
import pandas as pd
import sys
import os

from qpso_strategy_finder import QPSOStrategyFinder

sys.path.insert(0, '../etf_data')
from etf_data_loader import load_all_data_from_file

import matplotlib.pyplot as plt

start_date = '2010-01-01'
end_date = '2018-04-08'

prefix = 'mil_'


def load_data():
    df_adj_close = load_all_data_from_file(prefix + 'etf_data_adj_close.csv', start_date, end_date)
    ticket_list = load_ticket_list(df_adj_close)
    print('cleaning data...')
    df_adj_close = df_adj_close[ticket_list]
    print('Backfill of data for NaNs...')
    df_adj_close = df_adj_close.fillna(method='bfill')
    print('Cleaning anomalies from data...')
    for ticket in ticket_list:
        pct = df_adj_close[ticket].pct_change()
        for i in range(len(df_adj_close[ticket])):
            if i > 0 and i < len(df_adj_close[ticket]) - 1:
                bound_lower = pct.iloc[i]
                bound_upper = pct.iloc[i + 1]
                # print('%d|%f|%f'%(i,bound_lower,bound_upper))
                if np.abs(bound_lower) > 0.1 and np.abs(bound_upper) > 0.1:
                    # print('changing price from %f to %f' %(df_adj_close[ticket].iloc[i], df_adj_close[ticket].iloc[i+1]))
                    df_adj_close[ticket].iloc[i] = df_adj_close[ticket].iloc[i + 1]

    ticket_list = remove_anomalies_tickets(df_adj_close,ticket_list)
    df_adj_close = df_adj_close[ticket_list]

    return ticket_list, df_adj_close


def load_ticket_list(df_adj_close):
    with open('../etf_data/' + prefix + 'etfs.txt', 'r') as fd:
        etf_list = list(fd.read().splitlines())
    print('Creating ticket list...')
    ticket_list = set(etf_list)
    tickets_to_remove = []
    print('Removing bad tickets')
    for ticket in ticket_list:
        if ticket not in df_adj_close.keys():
            tickets_to_remove.append(ticket)
        elif len(df_adj_close[ticket].loc[np.isnan(df_adj_close[ticket])]) == len(df_adj_close[ticket]):
            tickets_to_remove.append(ticket)

    ticket_list = ticket_list - set(tickets_to_remove)
    ticket_list = list(ticket_list)
    return ticket_list

def remove_anomalies_tickets(df_adj_close, ticket_list):
    ticket_list = set(ticket_list)
    tickets_to_remove = []
    print('Removing anomalies tickets')
    for ticket in ticket_list:
        if df_adj_close[ticket].pct_change().max() > 0.1:
            tickets_to_remove.append(ticket)

    ticket_list = ticket_list - set(tickets_to_remove)
    ticket_list = list(ticket_list)
    return ticket_list


data_filename = prefix + 'data.csv'

if not os.path.isfile(data_filename):
    ticket_list, df_adj_close = load_data()
    df_adj_close.to_csv(data_filename, index=False)
else:
    print('Loading data from file %s' % data_filename)
    df_adj_close = pd.read_csv(data_filename)
    ticket_list = df_adj_close.keys()
    print(ticket_list)

# print('Running data smoothing with EWM...')
# df_adj_close = df_adj_close.ewm(alpha=0.55).mean()
print('Computing price changes')
price_change = df_adj_close.pct_change()

print('Running optimisation...')

strategyFinder = QPSOStrategyFinder(10, 100, m=len(ticket_list))
best = strategyFinder.run(price_change)

plt.plot(best.evaluate(price_change))
plt.show()

selected_tickets = []
selected_w = []
for i in range(len(ticket_list)):
    if best.w[i] > 0.1:
        print('%s:%f' % (ticket_list[i], best.w[i]))
        selected_tickets.append(ticket_list[i])
        selected_w.append(best.w[i])

print(np.sum(best.w))
print(selected_tickets)
print(selected_w)

plt.plot(df_adj_close[selected_tickets])
plt.show()

plt.plot(df_adj_close[selected_tickets].pct_change())
plt.show()
