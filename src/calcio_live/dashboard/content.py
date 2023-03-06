# from dashboard.layout.callbacks import butt_load_models
# from dashboard.layout.callbacks import callbacks2
from dashboard.index import app
from dashboard.layout.tab_predict import tab_predict
from dashboard.layout.tab_files import tab_files
from dashboard.layout.tab_load_models import tab_load_models
from dashboard.layout.callbacks import butt_load_models
from dashboard.layout.callbacks import butt_predict
from dash import html
import dash_bootstrap_components as dbc


app.layout = html.Div([
        dbc.Row(html.H4('Live Prediction Dashboard'),
                style={'margin-bottom': 40}),
        dbc.Tabs([
            dbc.Tab(tab_predict, label='Prediction'),
            dbc.Tab(tab_files, label='Files'),
            dbc.Tab(tab_load_models, label='Load Models')
        ]),
    ],
        style={'margin-left': '80px', 'margin-right': '80px'}
    )
