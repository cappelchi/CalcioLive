import click
from utils import console_predict_v2


@click.command()
@click.argument(
    'model_types',
    type = str,
    nargs = -1
)
def main(model_types:str):
    model_type_dict = {
        'mc':'FOOT-LIVEMC',
        'tt':'FOOT-LIVETOTAL',
        'hc':'FOOT-LIVEHCAP',
        'bc':'FOOT-LIVEBC',
        'bc2':'FOOT-LIVEBC2'
    }
    if model_types:
        model_type_list = [model_type_dict[arg] for arg in model_types]
    else:
        model_type_list = ['FOOT-LIVEMC', 'FOOT-LIVETOTAL', 'FOOT-LIVEHCAP']
    console_predict_v2(model_type_list)


if __name__ == "__main__":
    main()