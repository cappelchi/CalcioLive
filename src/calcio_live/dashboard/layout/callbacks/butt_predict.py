from utils import make_predict
from dashboard.index import app
from dash.dependencies import Input, Output
from time import gmtime, strftime, time


@app.callback(
    output= {'info':Output('predict-info-1', 'children')},
    inputs = {'predict_button':Input('predict_button', 'n_clicks')},
    prevent_initial_call=True,
    suppress_callback_exceptions=True
            )
def update_click(predict_button):
    start_time = time()
    make_predict()
    return {
        'info':f'''
        {strftime("%Y-%m-%d %H:%M:%S", gmtime())}: predict ready in _{time() - start_time}_sec
        '''
            }
