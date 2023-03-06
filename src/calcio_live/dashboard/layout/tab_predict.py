from dash import html
import dash_bootstrap_components as dbc

tab_predict = [
        dbc.Row([
            dbc.Col(
                dbc.Button('predict', id = 'predict_button',
                n_clicks = 0, className = 'mr-2')
                ),
            dbc.Col(
                html.Div('Запустить предикт')
                )
                ], style = {'margin-top':20, 'margin-bottom':40}),
        dbc.Row([
            dbc.Col(html.Div(id = 'predict-info-1')),
            dbc.Col(html.Div(id = 'predict-info-2'))
                ], style = {'margin-bottom':40})
            ]