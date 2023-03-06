from dash import dcc
from dash import html
import dash_bootstrap_components as dbc

tab_load_models = [
        ######################## выбор 1x2 ##############################
        dbc.Row(html.Div([
            html.Div('Тип ставки: 1Х2 | Модель | Версия'),
            dcc.Input(id='model-1-name', type='text', value='FOOT-LIVEMC'),
            dcc.Input(id='model-1-num', type='text', value=2),
        ], style={'margin-top': 5, 'margin-bottom': 5})
        ),
        ######################## выбор handicap ##########################
        dbc.Row(html.Div([
            html.Div('Тип ставки: HANDICAP | Модель n.1 | Версия'),
            dcc.Input(id='model-2-name', type='text', value='FOOT-LIVEBST1'),
            dcc.Input(id='model-2-num', type='text', value=3),
        ], style={'margin-top': 5, 'margin-bottom': 5})
        ),
        dbc.Row(html.Div([
            html.Div('Тип ставки: HANDICAP | Модель n.2 | Версия'),
            dcc.Input(id='model-3-name', type='text', value='FOOT-LIVEBST2'),
            dcc.Input(id='model-3-num', type='text', value=4),
        ], style={'margin-top': 5, 'margin-bottom': 5})
        ),
        ######################## выбор total #############################
        dbc.Row(html.Div([
            html.Div('Тип ставки: TOTAL | Модель n.1 | Версия'),
            dcc.Input(id='model-4-name', type='text', value='FOOT-LIVEBST1'),
            dcc.Input(id='model-4-num', type='text', value=3),
        ], style={'margin-top': 5, 'margin-bottom': 5})
        ),
        dbc.Row(html.Div([
            html.Div('Тип ставки: TOTAL | Модель n.2 | Версия'),
            dcc.Input(id='model-5-name', type='text', value='FOOT-LIVEBST2'),
            dcc.Input(id='model-5-num', type='text', value=4),
        ], style={'margin-top': 5, 'margin-bottom': 5})
        ),
        ######################## выбор total #############################
        dbc.Row([
            dbc.Col(
                dbc.Button('load', id='start_butt',
                           n_clicks=0, className='mr-2')
            )
        ], style={'margin-top': 20, 'margin-bottom': 40}),
        dbc.Row([
            html.Div(id='output-info')
        ], style={'margin-top': 40})
    ]