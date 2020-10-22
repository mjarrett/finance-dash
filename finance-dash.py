# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_auth
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash.dependencies import Input, Output
import dash_table
from dash.exceptions import PreventUpdate 

from credentials import *

import pandas as pd
import datetime as dt
import json
import numpy as np
import re

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




def make_dfs():
    today = dt.datetime.now()
    limit = today-dt.timedelta(30)
    limit = dt.datetime(limit.year,limit.month,1)
    start = dt.datetime(2019,1,1)

    start_str = start.strftime('%Y-%m-%d')
    limit_str = limit.strftime('%Y-%m-%d')
    year = today.strftime('%Y')

    #df_transactions = get_sheet(url,gid_expenses)
    #df_income = get_sheet(url,gid_income)
    df_assets = get_sheet(url,gid_assets).applymap(tofloat)

    df_housing = get_sheet(url,gid_housing)
    df_savings = get_sheet(url,gid_savings)
    df_visa_transactions = get_sheet(url,gid_visa_transactions)
    df_chequing_transactions = get_sheet(url,gid_chequing_transactions)
    df_visa_mike_transactions = get_sheet(url,gid_visa_mike_transactions)
    df_other_transactions = get_sheet(url,gid_other_transactions)

    transactions_dfs = [df_visa_transactions,df_chequing_transactions, df_visa_mike_transactions,df_other_transactions]

    df_transactions = pd.concat(transactions_dfs)
    df_transactions = df_transactions[~df_transactions.Category.isin(['Housing','Transfer'])].sort_index(ascending=False)
    df_expenses = make_table(df_transactions,value='Price')
    df_income = make_table(df_transactions,value='Refund')

    sdf = pd.concat([df_expenses[start:limit].sum(1),
                df_housing.sum(1),
                df_savings.sum(1),
                df_income.sum(1)],
                axis=1)
    sdf.columns = ['expenses','housing','savings','income']


    sdf = sdf.loc[start:limit].reset_index()
    sdf.columns = ['Date','expenses','housing','savings','income']
    
    
    sdf.to_csv('data/sdf.csv', index=False)
    df_transactions.to_csv('data/df_transactions.csv', index=True)
    df_expenses.to_csv('data/df_expenses.csv', index=True)
    df_income.to_csv('data/df_income.csv', index=True)
    df_housing.to_csv('data/df_housing.csv', index=True)
    df_assets.to_csv('data/df_assets.csv',index=True)
    return sdf, df_transactions, df_expenses, df_income, df_housing, df_assets
  
def get_dfs():
    
    try:
        sdf = pd.read_csv('data/sdf.csv', index_col='Date')
        sdf.index = pd.to_datetime(sdf.index)
        df_transactions = pd.read_csv('data/df_transactions.csv', index_col='Date')
        df_transactions.index = pd.to_datetime(df_transactions.index)
        df_expenses = pd.read_csv('data/df_expenses.csv', index_col='Month')
        df_expenses.index = pd.to_datetime(df_expenses.index)
        df_income = pd.read_csv('data/df_income.csv', index_col='Month')
        df_income.index = pd.to_datetime(df_income.index)
        df_housing = pd.read_csv('data/df_housing.csv', index_col='Date')
        df_housing.index = pd.to_datetime(df_housing.index)
        df_assets = pd.read_csv('data/df_assets.csv',index_col='Date')
        df_assets.index = pd.to_datetime(df_assets.index)
        
        return sdf, df_transactions, df_expenses, df_income, df_housing, df_assets
        
    except FileNotFoundError:
        return make_dfs()
        
    
    
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

def tofloat(x):
    if type(x) == float:
        return x
    x = re.sub("[^0-9\.]", "", x)
    return float(x)

def get_sheet(baseurl,gid,):
    url = f'{baseurl}/export?gid={gid}&format=csv'

    df = pd.read_csv(url,index_col='Date')
    df.index = pd.to_datetime(df.index)  

    return df



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'Home Finances'
#######
# AUTH : https://dash.plot.ly/authentication
auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)
#######


def make_datatable(df):
    df.index.name = 'Date'
    df = df.reset_index()
    df = df[['Date','Purchase','Price','Refund','Category']]

    table = dash_table.DataTable(
        id='transactions-table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.reset_index().to_dict('records'),
        editable=False,
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        column_selectable="single",
        row_selectable="multi",
        row_deletable=False,
        selected_columns=[],
        selected_rows=[],
        page_action="native",
        page_current= 0,
        #page_size= 10,
        )
    return table

 
    

def make_fig1():

    sdf, df_transactions, df_expenses, df_income, df_housing, df_assets = get_dfs()

    data = [
        go.Bar(
            x = sdf.index,
            y = sdf['expenses'],
            name='Expenses',
            marker_color=colors['Expenses']
            ),
        go.Bar(
            x = sdf.index,
            y = sdf['housing'],
            name='Housing costs',
            marker_color=colors['Housing']
            ),
        go.Bar(
            x = sdf.index,
            y = sdf['savings'],
            name='Savings',
            marker_color=colors['Savings']
            ),
        go.Scatter(
            x=sdf.index, 
            y=sdf['income'],
            name='Income',
            marker_color=colors['Income']
            )
        ]
    
    data2 = [

        go.Scatter(
            x=df_assets.index, 
            y=df_assets['Total Liquid'],
            name='Cash + TFSA',
            marker_color='#9467bd'
            ),
        go.Scatter(
            x=df_assets.index, 
            y=df_assets['Home value']-df_assets['Mortgage balance'],
            name='Home Equity',
            marker_color='#e377c2'
            ),
        go.Scatter(
            x=df_assets.index, 
            y=df_assets['Mike RRSP']+df_assets['Christa RRSP'],
            name='RRSP',
            marker_color='#17becf'
            ),
        
        ]
    
    fig = make_subplots(rows=2, cols=1, 
                    shared_xaxes=True, 
                    vertical_spacing=0.09,
                    subplot_titles=("Budget", "Assets"))
    fig.add_trace(data[0], row=1,col=1)
    fig.add_trace(data[1], row=1,col=1)
    fig.add_trace(data[2], row=1,col=1)
    fig.add_trace(data[3], row=1,col=1)
    
    fig.add_trace(data2[0], row=2,col=1)
    fig.add_trace(data2[1], row=2,col=1)
    fig.add_trace(data2[2], row=2,col=1)
    
    fig.update_layout(
        barmode = 'stack',
        dragmode='pan'
        #title = f'Budget'
        )
    fig.update_xaxes(range=[sdf.index[-12],sdf.index[-1]])
    fig.update_yaxes(fixedrange=True)

    return fig




def make_layout():
    today = dt.datetime.now()
    limit = today-dt.timedelta(30)
    limit = dt.datetime(limit.year,limit.month,1)
    start = dt.datetime(2019,1,1)

    start_str = start.strftime('%Y-%m-%d')
    limit_str = limit.strftime('%Y-%m-%d')
    year = today.strftime('%Y')
    
    
    #make_dfs()
    sdf, df_transactions, df_expenses, df_income, df_housing, df_assets = get_dfs()
    print(sdf.index)
    curr_savings = sdf.loc[limit_str, 'income'] - sdf.loc[limit_str, 'housing'] - sdf.loc[limit_str, 'expenses']
    ann_savings = sdf.loc[year,'income'].sum() - sdf.loc[year,'housing'].sum() - sdf.loc[year,'expenses'].sum()
    
    controls = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label("Category"),
                dcc.Dropdown(
                    id="category-dropdown",
                    options=[
                       {"label": col, "value": col} for col in list(set(df_transactions['Category'].dropna())) + ['All']
                    ],
                    value='All',
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Month"),
                dcc.Dropdown(
                    id="month-dropdown",
                    options=[{"label": col.strftime('%b %Y'), "value": col.strftime('%Y-%m')} for col in sdf.index] + [{'label':'All','value':'All'}],
                    value='All',
                ),
            ]
        ),

    ],
    body=True,
)
    
    
    layout =  dbc.Container(children=[
        html.H1(children='Houshold Finances'),
        
        dbc.Row([
            dbc.Col([

                html.A("Here\'s the raw spreadsheet", href=raw_url),
                html.Div(f"Savings this year: ${ann_savings:.2f}"),
                html.Div(f"Savings this month: ${curr_savings:.2f}"),
                ]),
            dbc.Col([
                dbc.Button("Refresh data", size="lg", className="mr-1", id='refresh-button')
            ])
        ]),
            
        
        dbc.Row([
        
            html.Div(id='current-cat-data',style={'display':'none'}),
            dcc.Graph(id='timeseries-graph',figure=make_fig1()),  
        ]),

        dbc.Row([
            dcc.Graph(id='detail-graph',style={'display': 'none'}),
            dcc.Graph(id='expense-timeseries',style={'display': 'none'}),
            dcc.Interval(
                id='interval-component',
                interval=10*1000, # in milliseconds
                n_intervals=0
            ),
        ]),
        
        dbc.Row([
            html.H3("Transactions"),
            controls,
            html.Div([make_datatable(df_transactions)])
        ])
    ])
    return layout

# wrap the layout in the content div so i can refresh everything with a callback
app.layout = html.Div([
    html.Div(id='content', children=make_layout())
])




# Refresh data
@app.callback(Output('content', 'children'),
              [Input('refresh-button','n_clicks')])
def refresh_data(n_clicks):
    print(n_clicks)
    if n_clicks is not None:
        make_dfs()
        return make_layout()
    
    else:
        raise PreventUpdate 

# Update datatable
@app.callback([Output('transactions-table','data')],
              [Input('category-dropdown','value'),Input('month-dropdown','value')])
def update_datatable(cat,month):
    print(cat,month, type(month))
    df = pd.read_csv('data/df_transactions.csv', index_col='Date')
    df.index = pd.to_datetime(df.index)
    
    df.index.name = 'Date'
    if month != 'All':
        df = df[month]
    df = df.reset_index()
    df = df[['Date','Purchase','Price','Refund','Category']]
    if cat != 'All':
        df = df[df['Category']==cat]
    
    data = df.reset_index().to_dict('records')
    
    return data,

@app.callback([Output('detail-graph','figure'), Output('detail-graph','style')],
              [Input('timeseries-graph','clickData')])
def make_detail_fig(clickData):
    
    if clickData is None:
        raise PreventUpdate
        
        
    sdf, df_transactions, df_expenses, df_income, df_housing, df_assets = get_dfs()
        
    date = clickData['points'][0]['x']
    cat  = clickData['points'][0]['curveNumber']
    
    
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
    
    sdf, df_transactions, df_expenses, df_income, df_housing, df_assets = get_dfs()
    
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
    app.run_server(debug=True,host='0.0.0.0',port=8050)
