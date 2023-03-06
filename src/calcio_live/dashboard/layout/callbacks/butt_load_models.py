from time import time
from utils import load_model
from dashboard.index import app
from dash.dependencies import Input, Output, State

@app.callback(
    output= {
        'info':Output('output-info', 'children')
            },
    inputs = {'start_button':Input('start_butt', 'n_clicks')},
    state = {
        'names':{cnt:State(f'model-{cnt}-name', 'value') for cnt in range(1, 6)},
        'nums':{cnt:State(f'model-{cnt}-num', 'value') for cnt in range(1, 6)},
            },
    prevent_initial_call=True,
    suppress_callback_exceptions=True
                )
def butt_load_models(start_button, names, nums):
    start_time = time()
    model_types = ['', '1x2:1', 'total:1', 'total:2', 'handicap:1', 'handicap:2']
    for nmod in range (1, 6):
        name = names[nmod]
        mnum = nums[nmod]
        model_type = model_types[nmod]
        _ = load_model(model_type, name, mnum)
    return {
        'info':f'''
        Download models:
        |1x2|: {names[1]} - {nums[1]}
        ***|HANDICAP m.1|: {names[2]} - {nums[2]}
        ***|HANDICAP m.2|: {names[3]} - {nums[3]}
        ***|TOTAL m.1|: {names[4]} - {nums[4]}
        ***|TOTAL m.2|: {names[5]} - {nums[5]} in _{time() - start_time}_sec
        '''
            }

