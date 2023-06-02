import click
import os
import yaml
from catboost import CatBoost
from utils import console_folded_predict


def features_list():
    return [
        "min_norm",
        "Score1_norm",
        "Score2_norm",
        "Score_diff",
        "Score_cat_1",
        "Score_cat_2",
        "Score_cat_3",
        "Score_cat_4",
        "Score_cat_5",
        "Score_cat_6",
        "Score_cat_7",
        "Score_cat_8",
        "Score_cat_9",
        "A1_scaled",
        "A2_scaled",
        "A1perMIN",
        "A2perMIN",
        "A1relativ",
        "A2relativ",
        "DA1_scaled",
        "DA2_scaled",
        "DA1perMIN",
        "DA2perMIN",
        "DA1relativ",
        "DA2relativ",
        "Pos1_cleaned",
        "Pos2_cleaned",
        "Off1_norm",
        "Off2_norm",
        "On1_norm",
        "On2_norm",
        "YC1_transformed",
        "YC2_transformed",
        "RC1_transformed",
        "RC2_transformed",
        "Sub1_transformed",
        "Sub2_transformed",
        "Cor1_transformed",
        "Cor2_transformed",
        "P1_transformed",
        "P2_transformed",
    ]


@click.command()
@click.option(
    "--multiclass",
    "-mc",
    default="A1relativ, A2relativ, DA1relativ, DA2relativ",
    help="features not used in multiclass separated by commas",
)
@click.option(
    "--total_over",
    "-th",
    default="A1relativ, A2relativ, DA1relativ, DA2relativ",
    help="features not used in total over separated by commas",
)
@click.option(
    "--total_under",
    "-ta",
    default="A1relativ, A2relativ, DA1relativ, DA2relativ",
    help="features not used in total under separated by commas",
)
@click.option(
    "--handicap_home",
    "-hh",
    default="A1relativ, A2relativ, DA1relativ, DA2relativ",
    help="features not used in handicap home separated by commas",
)
@click.option(
    "--handicap_away",
    "-ha",
    default="A1relativ, A2relativ, DA1relativ, DA2relativ",
    help="features not used in handicap away separated by commas",
)
@click.option(
    "--flip",
    "-fm",
    default=False,
    help="flip multiclass prediction for multiclass model n.2 only",
)
def prepare_predict(**params):
    """
    :return:
    1. Make dict of models for all types and folds via yaml
    2. Make features list
    """

    match_cols_list = features_list()
    yaml_path = os.path.join(os.path.abspath("./models"), "model_path.yml")
    with open(yaml_path, "r") as yml:
        models_info_dict = yaml.load(yml, Loader=yaml.FullLoader)
    models_dict = {}
    model_feature_dict = {}
    for model_type in models_info_dict.keys():
        models_dict[model_type] = {}
        not_used = params[model_type].replace(" ", "").split(",")
        model_feature_dict[model_type] = [
            True if item not in not_used else False for item in match_cols_list
        ]
        for cnt, fold_name in enumerate(models_info_dict[model_type]["folds_names"]):
            models_dict[model_type][cnt] = CatBoost().load_model(
                models_info_dict[model_type]["path_to_models"][fold_name]
            )

    console_folded_predict(models_dict, model_feature_dict, params["flip"])
    return


if __name__ == "__main__":
    prepare_predict()
