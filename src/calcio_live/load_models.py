import click
from utils import download_folded_models


@click.command()
@click.option('--multiclass', '-m', help='Version of multiclass model')
@click.option('--total', '-t', help='Version of total model')
@click.option('--handicap', '-hp', help='Version of handicap model')
@click.option('--binaryclassx2', '-bcx2', help='Version of binaryclass model 1 vs X2')
@click.option('--binaryclass1x', '-bc1x', help='Version of binaryclass model 1X vs 2')
def main(**params):
    model_type_dict = {
        'FOOT-LIVEMC': {'version':params['multiclass']},
        'FOOT-LIVETOTAL': {'version':params['total']},
        'FOOT-LIVEHCAP': {'version':params['handicap']},
        'FOOT-LIVEBC': {'version': params['binaryclassx2']},
        'FOOT-LIVEBC2': {'version': params['binaryclass1x']}
    }
    download_folded_models(**model_type_dict)


if __name__ == "__main__":
    main()
