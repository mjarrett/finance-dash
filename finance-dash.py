# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_auth
import plotly.graph_objects as go
import plotly.express as px
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
    
    print('Querying google doc')

    today = dt.datetime.now()
    limit = today-dt.timedelta(30)
    limit = dt.datetime(limit.year,limit.month,1)
    start = dt.datetime(2019,1,1)

    start_str = start.strftime('%Y-%m-%d')
    limit_str = limit.strftime('%Y-%m-%d')
    year = today.strftime('%Y')




    df_visa_transactions = get_sheet(url,gid_visa_transactions)
    df_chequing_transactions = get_sheet(url,gid_chequing_transactions)
    df_visa_mike_transactions = get_sheet(url,gid_visa_mike_transactions)
    df_other_transactions = get_sheet(url,gid_other_transactions)

    
    
    transactions_dfs = [df_visa_transactions,df_chequing_transactions, df_visa_mike_transactions,df_other_transactions]

    df_transactions = pd.concat(transactions_dfs)
    df_transactions = df_transactions[~df_transactions['Label'].isin(['Transfer'])].sort_index(ascending=False)

    
     
    xw = df_visa_transactions[['Labels','Categories']].dropna().set_index('Labels')['Categories'].to_dict()  
    df_transactions['Category'] = df_transactions['Label'].map(xw)
    df_transactions['Net'] = df_transactions.fillna(0)['Price'] - df_transactions.fillna(0)['Refund']
    df_transactions['Month'] = pd.to_datetime(df_transactions.index.strftime('%Y-%m'))
    df_monthly = df_transactions.pivot_table(values='Net',index='Month',columns=['Category', 'Label'],aggfunc='sum')
    
    # add assets to monthly df
    df_assets = get_sheet(url,gid_assets).applymap(tofloat)
    cols = pd.MultiIndex.from_tuples([('Assets',x) for x in df_assets.columns], names=['Category', 'Label'])
    df_monthly = pd.concat([df_monthly,pd.DataFrame(df_assets.values,columns=cols, index=df_assets.index)], axis=1)
    
    df_monthly.to_csv('data/df_monthly.csv')
    df_transactions.to_csv('data/df_transactions.csv', index=True)
    
    return df_monthly, df_transactions
  
def get_dfs():
    
    try:
        df_monthly = pd.read_csv('data/df_monthly.csv', header=[0,1], index_col=0)
        df_monthly.index = pd.to_datetime(df_monthly.index)
        df_transactions = pd.read_csv('data/df_transactions.csv', index_col='Date')
        df_transactions.index = pd.to_datetime(df_transactions.index)
        
        return df_monthly, df_transactions
        
    except FileNotFoundError:
        return make_dfs()
    
    
    

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

make_dfs()

external_stylesheets = [dbc.themes.BOOTSTRAP]

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
    df = df[['Date','Purchase','Price','Refund','Label']]

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

    df_monthly, df_transactions = get_dfs()
    df_expenses = df_monthly.loc[:,~df_monthly.columns.get_level_values(0).isin(['Housing','Income','Savings','Assets'])].sum(1)
    df_housing = df_monthly['Housing'].sum(1)
    df_savings = df_monthly['Savings'].sum(1)
    df_income = df_monthly['Income'].sum(1)
    df_assets = df_monthly['Assets']
    
    data = [
        go.Bar(
            x = df_expenses.index,
            y = df_expenses.values,
            name='Expenses',
            marker_color=colors['Expenses']
            ),
        go.Bar(
            x = df_housing.index,
            y = df_housing.values,
            name='Housing costs',
            marker_color=colors['Housing']
            ),
        go.Bar(
            x = df_savings.index,
            y = df_savings.values,
            name='Savings',
            marker_color=colors['Savings']
            ),
        go.Scatter(
            x=df_income.index[:-1], 
            y=df_income.abs().values[:-1],
            name='Income',
            marker_color=colors['Income'],
            mode='lines+markers'
            )
        ]
    
    data2 = [

        go.Scatter(
            x=df_assets.index, 
            y=df_assets['Total Liquid'],
            name='Cash + TFSA',
            marker_color='#9467bd',
            mode='lines+markers'
            ),
        go.Scatter(
            x=df_assets.index, 
            y=df_assets['Home value']-df_assets['Mortgage balance'],
            name='Home Equity',
            marker_color='#e377c2',
            mode='lines+markers'
            ),
        go.Scatter(
            x=df_assets.index, 
            y=df_assets['Mike RRSP']+df_assets['Christa RRSP'],
            name='RRSP',
            marker_color='#17becf',
            mode='lines+markers'
            ),
        
        ]
    
    data3 = [
        go.Bar(
            x=df_income.index,
            y=-df_income-df_housing-df_expenses,
            name='Cash flow',
            marker_color='#bcbd22'
        )
    ]
    
    fig = make_subplots(rows=3, cols=1, 
                    shared_xaxes=True, 
                    vertical_spacing=0.09,
                    subplot_titles=("Budget", "Assets", "Cash flow"))
    fig.add_trace(data[0], row=1,col=1)
    fig.add_trace(data[1], row=1,col=1)
    fig.add_trace(data[2], row=1,col=1)
    fig.add_trace(data[3], row=1,col=1)
    
    fig.add_trace(data2[0], row=2,col=1)
    fig.add_trace(data2[1], row=2,col=1)
    fig.add_trace(data2[2], row=2,col=1)
    
    fig.add_trace(data3[0], row=3, col=1)
    
    fig.update_layout(
        barmode = 'stack',
        dragmode='pan',
        height=700, #px
        #title = f'Budget'
        )
    fig.update_xaxes(range=[df_monthly.index[-12],df_monthly.index[-1]])
    fig.update_yaxes(fixedrange=True)

    return fig

def make_expenses_fig(date_str='All'): 
    df_monthly, df_transactions = get_dfs()
    df = df_monthly.loc[:,~df_monthly.columns.get_level_values(0).isin(['Housing','Income','Savings'])]

    if date_str == 'All':
        fig = go.Figure(data=[
            go.Bar(name='12 month average', y=df.sum(level=0, axis=1).mean(0), x=df.sum(level=0, axis=1).mean(0).index),
            ])
    else:
        date_str = date_str + '-01'
        fig = go.Figure(data=[
            go.Bar(name='12 month average', y=df.sum(level=0, axis=1).mean(0), x=df.sum(level=0, axis=1).mean(0).index),
            go.Bar(name=pd.to_datetime(date_str).strftime('%b %Y'), y=df.sum(level=0, axis=1).loc[date_str], x=df.sum(level=0, axis=1).loc[date_str].index)
            ])
    # Change the bar mode
    fig.update_layout(barmode='group')
    return fig


def make_cat_detail_fig(date_str,cat):

#     if cat == 'All':
#         return None
    
#     if date_str == 'All':
#         date_str = None


    

    if cat == 'All':
        data = go.Scatter(y=[1,2,3], x=[1,2,3])
        fig = go.Figure(data=data)
    else:
        df_monthly, df_transactions = get_dfs()
        df = df_monthly.loc[date_str:date_str]#,~df_monthly.columns.get_level_values(0).isin(['Housing','Income','Savings'])]
        data = [go.Scatter(name=y,y=df.loc[:,cat].iloc[:,x], x=df.index) for x,y in 
                          zip(range(len(df.loc[:,cat].columns)), df.loc[:,cat].columns)]
        fig = px.area(df.loc[:,cat])

    
    return fig
            
    
    
def make_layout():
    today = dt.datetime.now()
    limit = today-dt.timedelta(30)
    limit = dt.datetime(limit.year,limit.month,1)
    start = dt.datetime(2019,1,1)
 
    start_str = start.strftime('%Y-%m-%d')
    limit_str = limit.strftime('%Y-%m-%d')
    year = today.strftime('%Y')
    
    
    make_dfs()
    df_monthly, df_transactions = get_dfs()
    df_expenses = df_monthly.loc[:,~df_monthly.columns.get_level_values(0).isin(['Housing','Income','Savings','Assets'])].sum(1)
    df_housing = df_monthly['Housing'].sum(1)
    df_savings = df_monthly['Savings'].sum(1)
    df_income = df_monthly['Income'].sum(1)
    df_cashflow = -df_income-df_housing-df_expenses
    curr_savings = df_cashflow.iloc[-2]
    ann_savings = df_cashflow.iloc[-13:-1].sum()
    controls = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label("Category"),
                dcc.Dropdown(
                    id="category-dropdown",
                    options=[
                       {"label": col, "value": col} for col in ['All'] + sorted(set(df_transactions['Category'].dropna()))
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
                    options=[{"label": m.strftime('%b %Y'), "value": m.strftime('%Y-%m')} for m in df_monthly.index] + [{'label':'All','value':'All'}],
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
            ], width=12),
            dbc.Col([
                dbc.Button("Refresh data", size="md", className="mr-1", id='refresh-button'),
            ], width=12),

        ]),
        dbc.Tabs([
            dbc.Tab(label='Overview', children=[
 
          
                html.Div(f"Avg monthly savings last 12 months: ${ann_savings/12:,.2f}"),
                html.Div(f"Savings last month: ${curr_savings:,.2f}"),

                dbc.Row([
                    dbc.Col([
                    html.Div(id='current-cat-data',style={'display':'none'}),
                    dcc.Graph(id='timeseries-graph',figure=make_fig1()),  
                    ], width=12),
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
            ]),
            
            dbc.Tab(label='Expenses', children=[
        
                dbc.Row([
                    html.H3("Transactions"),
                    dbc.Col([
                        controls
                    ], width=12),
                    dbc.Col([
                        dcc.Graph(id='expenses-avg', figure=make_expenses_fig()),
                    ], width=12),
                    dbc.Col([
                        html.Div(id='category-detail-div', className='d-none', children=[
                            dcc.Graph(id='category-detail'),# figure=make_cat_detail_fig(None,'Food')),
                        ]),
                    ], width=12),

                    dbc.Col([
                        make_datatable(df_transactions)
                    ], width=12)
                ])
            ]),
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
    if n_clicks is not None:
        make_dfs()
        return make_layout()
    
    else:
        raise PreventUpdate 

# Update datatable
@app.callback([Output('transactions-table','data'), Output('category-detail', 'figure'), Output('expenses-avg', 'figure'), 
                Output('category-detail-div', 'className')],
              [Input('category-dropdown','value'),Input('month-dropdown','value')])
def update_datatable(cat,month):
    df_monthly, df_transactions = get_dfs()
    
    df = df_transactions.copy()
    if month != 'All':
        df = df[month]
    
    if cat != 'All':
        df = df[df['Category']==cat]

    df = df.reset_index()   
    data = df[['Date','Purchase','Price','Refund','Label']].reset_index().to_dict('records')
    className = 'd-none' if cat=='All' else ''
    return data,make_cat_detail_fig(None,cat), make_expenses_fig(month), className

# @app.callback([Output('detail-graph','figure'), Output('detail-graph','style')],
#               [Input('timeseries-graph','clickData')])
# def make_detail_fig(clickData):
    
#     if clickData is None:
#         raise PreventUpdate
        
        
#     sdf, df_transactions, df_expenses, df_income, df_housing, df_assets = get_dfs()
        
#     date = clickData['points'][0]['x']
#     cat  = clickData['points'][0]['curveNumber']
    
    
#     if cat == 0:
#         category = 'Expenses'
#         df = df_expenses.loc[date]
#     elif cat == 1:
#         category = 'Housing'
#         df = df_housing.loc[date]
#     elif cat == 2:
#         category = 'Savings'
#         df = df_savings.loc[date]
#     elif cat == 3:
#         category = 'Income'
#         df = df_income.loc[date]
        
#     df = df.reset_index()
#     df.columns = ['categories','money']
#     data = [
#         go.Bar(
#             x = df['categories'],
#             y = df['money'],
#             name='Expenses',
#             marker_color=colors[category]
#             )
#         ]
    
#     layout = go.Layout(title = f'{category}')  
#     fig = go.Figure(data=data,layout=layout)
    
#     return fig, {'display': 'inline'}

# @app.callback([Output('expense-timeseries','figure'), Output('expense-timeseries','style')],
#               [Input('detail-graph','clickData'),Input('detail-graph','figure')])
# def make_exp_fig(clickData,fig):
    
#     sdf, df_transactions, df_expenses, df_income, df_housing, df_assets = get_dfs()
    
#     subcat = clickData['points'][0]['x']
#     category = fig['layout']['title']['text']  
    
    
    
#     if category == 'Expenses':
#         df = df_expenses.loc[:,subcat]
#     elif category == 'Housing':
#         df = df_housing.loc[:,subcat]
#     elif category == 'Savings':
#         df = df_savings.loc[:,subcat]
#     elif category == 'Income':
#         df = df_income.loc[:,subcat]
    
#     df = df.reset_index()
#     df.columns = ['Date','cost']
#     data = [
#         go.Scatter(
#             x=df['Date'], 
#             y=df['cost'],
#             name=subcat,
#             marker_color=colors[category]
#             )
#         ]
    
#     layout = go.Layout(title = f'{subcat}')
#     fig = go.Figure(data=data,layout=layout)
#     return fig, {'display': 'inline'}


if __name__ == '__main__':
    app.run_server(debug=True,host='0.0.0.0',port=8050)
