# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_auth
import plotly.graph_objects as go
from dash.dependencies import Input, Output

from credentials import *

import pandas as pd
import datetime
import json
import numpy as np

# colors = [
#     '#1f77b4',  # muted blue
#     '#ff7f0e',  # safety orange
#     '#2ca02c',  # cooked asparagus green
#     '#d62728',  # brick red
#     '#9467bd',  # muted purple
#     '#8c564b',  # chestnut brown
#     '#e377c2',  # raspberry yogurt pink
#     '#7f7f7f',  # middle gray
#     '#bcbd22',  # curry yellow-green
#     '#17becf'   # blue-teal
# ]
colors = {'Expenses':'#1f77b4', 
          'Housing':'#ff7f0e',
          'Savings': '#2ca02c',
          'Income':'#d62728'
         }


today = datetime.datetime.now()
limit = today-datetime.timedelta(30)
start = datetime.datetime(2019,1,1)

def make_table(df,value='Price'):

    df.index = pd.to_datetime(df.index)
    df = df.reset_index()


    df['Month'] = df['Date'].map(lambda x: x.strftime("%Y-%m"))

    df.Price = df.Price.astype(float)
    df.Refund = df.Refund.astype(float)


    table =  pd.pivot_table(df,values=value,columns=['Category'],index='Month',aggfunc=np.sum)
    table.index = pd.to_datetime(table.index)
    df.index.names = ['Date']
    return table
    
print('Querying google doc')
def get_sheet(baseurl,gid,):
    url = f'{baseurl}/export?gid={gid}&format=csv'

    df = pd.read_csv(url,index_col='Date')
    df.index = pd.to_datetime(df.index)

    return df

#df_transactions = get_sheet(url,gid_expenses)
#df_income = get_sheet(url,gid_income)
df_housing = get_sheet(url,gid_housing)
df_savings = get_sheet(url,gid_savings)
df_visa_transactions = get_sheet(url,gid_visa_transactions)
df_chequing_transactions = get_sheet(url,gid_chequing_transactions)
df_visa_mike_transactions = get_sheet(url,gid_visa_mike_transactions)
df_other_transactions = get_sheet(url,gid_other_transactions)

transactions_dfs = [df_visa_transactions,df_chequing_transactions, df_visa_mike_transactions,df_other_transactions]

df_transactions = pd.concat(transactions_dfs)
df_transactions = df_transactions[~df_transactions.Category.isin(['Housing','Transfer'])]
df_expenses = make_table(df_transactions,value='Price')
df_income = make_table(df_transactions,value='Refund')



#print(df_expense_table)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#######
# AUTH : https://dash.plot.ly/authentication
auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)
#######


def make_fig1():
    df = pd.concat([df_expenses[start:limit].sum(1),
                df_housing.sum(1),
                df_savings.sum(1),
                df_income.sum(1)],
                axis=1)
    df = df.loc[start:limit].reset_index()
    df.columns = ['Date','expenses','housing','savings','income']
    

    data = [
        go.Bar(
            x = df['Date'],
            y = df['expenses'],
            name='Expenses',
            marker_color=colors['Expenses']
            ),
        go.Bar(
            x = df['Date'],
            y = df['housing'],
            name='Housing costs',
            marker_color=colors['Housing']
            ),
        go.Bar(
            x = df['Date'],
            y = df['savings'],
            name='Savings',
            marker_color=colors['Savings']
            ),
        go.Scatter(
            x=df['Date'], 
            y=df['income'],
            name='Income',
            marker_color=colors['Income']
            )
        ]

    layout = go.Layout(
        barmode = 'stack',
        title = f'Budget'
        )
    fig1 = go.Figure(data=data,layout=layout)
    return fig1




app.layout = html.Div(children=[
        html.H1(children='Houshold Finances'),

        html.A("Here\'s the raw spreadsheet", href=raw_url),

        html.Div(id='current-cat-data',style={'display':'none'}),
        dcc.Graph(id='timeseries-graph',figure=make_fig1()),   

        dcc.Graph(id='detail-graph',style={'display': 'none'}),
        dcc.Graph(id='expense-timeseries',style={'display': 'none'}),
        dcc.Interval(
            id='interval-component',
            interval=10*1000, # in milliseconds
            n_intervals=0
        )
    ])




@app.callback([Output('detail-graph','figure'), Output('detail-graph','style')],
              [Input('timeseries-graph','clickData')])
def make_detail_fig(clickData):
    print(f"clickData: {clickData}")
    date = clickData['points'][0]['x']
    cat  = clickData['points'][0]['curveNumber']
    print(date)
    
    
    if cat == 0:
        category = 'Expenses'
        df = df_expenses.loc[date]
    elif cat == 1:
        category = 'Housing'
        df = df_housing.loc[date]
    elif cat == 2:
        category = 'Savings'
        df = df_savings.loc[date]
    elif cat == 3:
        category = 'Income'
        df = df_income.loc[date]
        
    df = df.reset_index()
    df.columns = ['categories','money']
    data = [
        go.Bar(
            x = df['categories'],
            y = df['money'],
            name='Expenses',
            marker_color=colors[category]
            )
        ]
    
    layout = go.Layout(title = f'{category}')  
    fig = go.Figure(data=data,layout=layout)
    
    return fig, {'display': 'inline'}

@app.callback([Output('expense-timeseries','figure'), Output('expense-timeseries','style')],
              [Input('detail-graph','clickData'),Input('detail-graph','figure')])
def make_exp_fig(clickData,fig):
    print(clickData)
    subcat = clickData['points'][0]['x']
    category = fig['layout']['title']['text']  
    print(subcat)
    
    
    
    if category == 'Expenses':
        df = df_expenses.loc[:,subcat]
        print(df.head())
    elif category == 'Housing':
        df = df_housing.loc[:,subcat]
    elif category == 'Savings':
        df = df_savings.loc[:,subcat]
    elif category == 'Income':
        df = df_income.loc[:,subcat]
    
    df = df.reset_index()
    df.columns = ['Date','cost']
    data = [
        go.Scatter(
            x=df['Date'], 
            y=df['cost'],
            name=subcat,
            marker_color=colors[category]
            )
        ]
    
    layout = go.Layout(title = f'{subcat}')
    fig = go.Figure(data=data,layout=layout)
    return fig, {'display': 'inline'}


if __name__ == '__main__':
    app.run_server(debug=False,host='0.0.0.0',port=8050)
