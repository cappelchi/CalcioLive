import click
import yaml
import os
from copy import deepcopy
from utils import check_folds
from utils import load_folded_model
from utils import load_model


models_info_dict = {
    "multiclass": {"name": "FOOT-LIVEMC", "model_type": "1x2"},
    "total_over": {"name": "FOOT-LIVEBST1", "model_type": "total"},
    "total_under": {"name": "FOOT-LIVEBST2", "model_type": "total"},
    "handicap_home": {"name": "FOOT-LIVEBST1", "model_type": "handicap"},
    "handicap_away": {"name": "FOOT-LIVEBST2", "model_type": "handicap"},
}


@click.command()
@click.option("--multiclass", "-mc", default="6", help="multiclass model version")
@click.option(
    "--total_over",
    "-th",
    default="6",
    help="total model version for over goals quantity prediction",
)
@click.option(
    "--total_under",
    "-ta",
    default="5",
    help="total model version for under goals quantity prediction",
)
@click.option(
    "--handicap_home",
    "-hh",
    default="6",
    help="handicap model version for away team goal quantity prediction",
)
@click.option(
    "--handicap_away",
    "-ha",
    default="5",
    help="handicap model version for away team goal quantity prediction",
)
def main(**params):
    fpd = deepcopy(models_info_dict)
    for model_type, model_features in fpd.items():
        fold_quantity, folds_tuple = check_folds(
            models_info_dict[model_type]["name"], params[model_type]
        )
        if fold_quantity:
            models_info_dict[model_type]["folds_quantity"] = fold_quantity
            models_info_dict[model_type]["folds_names"] = folds_tuple
            models_info_dict[model_type]["path_to_models"] = load_folded_model(
                model_type,
                models_info_dict[model_type]["name"],
                params[model_type],
                folds_tuple,
            )
        else:
            models_info_dict[model_type]["folds_quantity"] = fold_quantity
            models_info_dict[model_type]["folds_names"] = "0"
            models_info_dict[model_type]["path_to_models"] = {
                "0": load_model(
                    model_type, models_info_dict[model_type]["name"], params[model_type]
                )
            }

    fold_name = models_info_dict[model_type]["folds_names"][0]
    model_path = models_info_dict[model_type]["path_to_models"][fold_name]

    yaml_path = os.path.join(os.path.dirname(model_path), "model_path.yml")
    with open(yaml_path, "w") as yml:
        yaml.dump(models_info_dict, yml)


if __name__ == "__main__":
    main()
