import pandas as pd
import numpy as np
import os
import yaml
import neptune
import logging
from os.path import abspath
from glob import glob
from catboost import CatBoost
from scipy.stats import poisson


INPUT_DIR = abspath("./")
MODEL_DICT = {}

DATA_TYPES_DICT = {
    "Id": np.int32,
    "StatTime": np.datetime64,
    "Minute": np.int8,
    "Active": np.int8,
    "Score1": np.int8,
    "Score2": np.int8,
    "A1": np.int16,
    "A2": np.int16,
    "DA1": np.int16,
    "DA2": np.int16,
    "Pos1": np.float32,
    "Pos2": np.float32,
    "Off1": np.int8,
    "Off2": np.int8,
    "On1": np.int8,
    "On2": np.int8,
    "YC1": np.int8,
    "YC2": np.int8,
    "RC1": np.int8,
    "RC2": np.int8,
    "Sub1": np.int8,
    "Sub2": np.int8,
    "Pen1": np.int8,
    "Pen2": np.int8,
    "Cor1": np.int8,
    "Cor2": np.int8,
    "Period": np.int8,
    "D": np.datetime64,
    "I": np.int32,
    "Active.1": np.int8,
    "Time": np.datetime64,
    "Minute.1": np.int8,
    "RawTime": np.datetime64,
    "Score1.1": np.int8,
    "Score2.1": np.int8,
    "Period.1": np.int8,
    "W1": np.float16,
    "WX": np.float16,
    "W2": np.float16,
    "X1": np.float16,
    "X2": np.float16,
    "W12": np.float16,
    "TotalValue": np.float16,
    "Over": np.float16,
    "Under": np.float16,
    "Hand1Value": np.float16,
    "H1": np.float16,
    "H2": np.float16,
}

COLS = [
    "StatTime",
    "Minute",
    "Active",
    "Score1",
    "Score2",
    "A1",
    "A2",
    "DA1",
    "DA2",
    "Pos1",
    "Pos2",
    "Off1",
    "Off2",
    "On1",
    "On2",
    "YC1",
    "YC2",
    "RC1",
    "RC2",
    "Sub1",
    "Sub2",
    "Pen1",
    "Pen2",
    "Cor1",
    "Cor2",
    "Period",
    "Comment",
    'vuoto'
]

USECOLS = [
    "Minute",
    "Active",
    "Score1",
    "Score2",
    "A1",
    "A2",
    "DA1",
    "DA2",
    "Pos1",
    "Pos2",
    "Off1",
    "Off2",
    "On1",
    "On2",
    "YC1",
    "YC2",
    "RC1",
    "RC2",
    "Sub1",
    "Sub2",
    "Pen1",
    "Pen2",
    "Cor1",
    "Cor2",
]


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


def get_credential(frmwork="neptune_team"):
    token_path = os.path.realpath("../credential.txt")
    with open(token_path, "r") as container:
        for line in container:
            if frmwork in line:
                login, psw = line.split(" ")[1], line.split(" ")[2].split("\n")[0]
                return login, psw


def neptune_download(model_name, model_num, path_to_model, metapath="model"):
    """
    Download neptune model by name and number to path
    no return
    :param model_num:
    :param model_name:
    :param path_to_model:
    :param metapath:
    """
    _, api_key = get_credential()
    model_version = neptune.init_model_version(
        project="scomesse/football",
        model=model_name,
        api_token=api_key,
        with_id=model_name + "-" + str(model_num),
    )

    try:
        model_version[metapath].download(path_to_model)
        model_version.stop()
        logging.info(f"Downloaded: {path_to_model}")
    except Exception:
        logging.error(f"Downloaded neptune: {path_to_model}")


def check_folds(model_name: str, model_num: str):
    """
    check if model has sub models for data folds
    :param model_name: FOOT-LIVEMC
    :param model_num: 3
    :return:
    """
    _, api_key = get_credential()
    model_version = neptune.init_model_version(
        project="scomesse/football",
        model=model_name,
        api_token=api_key,
        with_id=model_name + "-" + str(model_num),
    )
    model_structure_dict = model_version.get_structure()
    if "model" in model_structure_dict:
        model_version.stop()
        return False, ()
    elif "models" in model_structure_dict:
        folds_list = []
        folds_quantity = 0
        for key, value in model_structure_dict["models"].items():
            if type(value) == neptune.attributes.atoms.file.File:
                folds_quantity += 1
                folds_list.append(key)
        model_version.stop()
        return folds_quantity, tuple(folds_list)
    else:
        "error"


def load_model(model_type: str, model_name: str, num: str):
    """
    :param model_type: 'multiclass', 'total_over', 'total_under', 'handicap_home', 'handicap_away'
    :param model_name: FOOT-LIVEMC
    :param num: 3
    :return: local model path
    """
    model_num = str(num)
    models_dir = os.path.join(INPUT_DIR, "models")
    PATH_TO_MODEL = os.path.join(
        models_dir, f"booster_{model_type}_{model_name}_{model_num}.model"
    )
    neptune_download(model_name, model_num, PATH_TO_MODEL)
    MODEL_DICT[model_type] = CatBoost()
    MODEL_DICT[model_type].load_model(PATH_TO_MODEL)
    return PATH_TO_MODEL


def load_folded_model(model_type: str, model_name: str, num: str, folds_tuple: tuple):
    """
    Download model by folds and return dict of paths
    :param model_type: multiclass, total, handicap
    :param model_name: FOOT-LIVEMC
    :param num: 3
    :param folds_tuple: tuple
    :return: local model path
    """

    model_num: str = str(num)
    models_dir = os.path.join(INPUT_DIR, "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    path_to_model_dict = {}
    for fold in folds_tuple:
        fold_num: str = fold.split("_")[-1]
        PATH_TO_MODEL = os.path.join(
            models_dir,
            f"booster_{model_type}_{model_name}_{model_num}_{fold_num}.model",
        )
        neptune_download(
            model_name, model_num, PATH_TO_MODEL, metapath=f"/models/{fold}"
        )
        path_to_model_dict[fold] = PATH_TO_MODEL
    return path_to_model_dict


def create_predict_vector(file_path: str, match_cols: list):
    match_df = pd.read_csv(
        file_path,
        sep=";",
        names=COLS,
        skiprows=1,
        usecols=USECOLS,
        dtype=DATA_TYPES_DICT,
    )
    match_df.iloc[0, :] = match_df.iloc[0, :].fillna(0)
    match_df = match_df.fillna(method="ffill")
    P1, PX, P2 = pd.read_csv(
        file_path,
        sep=";",
        nrows=1,
        header=None,
        dtype={0: np.float32, 1: np.float32, 2: np.float32},
    ).values[0]
    match_df[["P1", "P2"]] = P1, P2
    match_df["min_norm"] = match_df["Minute"].astype(np.float32) / 50
    # трансформируем голы
    match_df[match_df["Score1"].isna()] = 0
    match_df["Score1_norm"] = (
        match_df["Score1"].fillna(method="ffill").fillna(0).astype(np.float32) / 4
    )
    match_df.loc[match_df["Score1"] > 3, ["Score1_norm"]] = 1.0
    match_df[match_df["Score2"].isna()] = 0
    match_df["Score2_norm"] = (
        match_df["Score2"].fillna(method="ffill").fillna(0).astype(np.float32) / 4
    )
    match_df.loc[match_df["Score2"] > 3, ["Score2_norm"]] = 1.0
    match_df["Score_diff"] = match_df["Score1"].astype(np.int16) - match_df[
        "Score2"
    ].astype(np.int16)
    match_df.loc[match_df["Score_diff"] < -4, ["Score_diff"]] = -4
    match_df.loc[match_df["Score_diff"] > 4, ["Score_diff"]] = 4
    match_df[[f"Score_cat_{n}" for n in range(1, 10)]] = np.eye(9)[
        match_df["Score_diff"].values + 4
    ]
    match_df["Score_diff"] = match_df["Score_diff"].astype(np.float32) / np.float32(4.0)
    # трансформируем атаки
    match_df["A1_scaled"] = match_df["A1"].astype(np.float32) / 75
    match_df.loc[match_df["A1"] >= 60, ["A1_scaled"]] = (
        60 + (match_df["A1"] - 60) / 4
    ) / 75
    match_df["A2_scaled"] = match_df["A2"].astype(np.float32) / 75
    match_df.loc[match_df["A2"] >= 60, ["A2_scaled"]] = (
        60 + (match_df["A2"] - 60) / 4
    ) / 75
    # атаки в минуту
    match_df["A1perMIN"] = match_df["A1"].astype(np.float32) / match_df[
        "Minute"
    ].astype(np.float32)
    match_df.loc[match_df["A1perMIN"] > 4, ["A1perMIN"]] = np.float32(4.0)
    match_df["A2perMIN"] = match_df["A2"].astype(np.float32) / match_df[
        "Minute"
    ].astype(np.float32)
    match_df.loc[match_df["A2perMIN"] > 4, ["A2perMIN"]] = np.float32(4.0)
    # динамика атак
    match_df["A1relativ"] = (
        match_df["A1"].astype(np.float32) - match_df["A1"].shift(5).astype(np.float32)
    ).fillna(0)
    match_df.loc[match_df["A1relativ"] > 15, ["A1relativ"]] = np.float32(15.0)
    match_df["A2relativ"] = (
        match_df["A2"].astype(np.float32) - match_df["A2"].shift(5).astype(np.float32)
    ).fillna(0)
    match_df.loc[match_df["A2relativ"] > 15, ["A2relativ"]] = np.float32(15.0)
    # трансформируем опасные атаки
    match_df["DA1_scaled"] = match_df["DA1"].astype(np.float32) / 50
    match_df.loc[match_df["DA1"] >= 40, ["DA1_scaled"]] = (
        80 + (match_df["DA1"] - 40) / 3
    ) / 100
    match_df["DA2_scaled"] = match_df["DA2"].astype(np.float32) / 50
    match_df.loc[match_df["DA2"] >= 40, ["DA2_scaled"]] = (
        80 + (match_df["DA2"] - 40) / 3
    ) / 100
    # опасные атаки в минуту
    match_df["DA1perMIN"] = match_df["DA1"].astype(np.float32) / match_df[
        "Minute"
    ].astype(np.float32)
    match_df.loc[match_df["DA1perMIN"] > 3, ["DA1perMIN"]] = np.float32(3.0)
    match_df["DA2perMIN"] = match_df["DA2"].astype(np.float32) / match_df[
        "Minute"
    ].astype(np.float32)
    match_df.loc[match_df["DA2perMIN"] > 3, ["DA2perMIN"]] = np.float32(3.0)
    # динамика опасных атак
    match_df["DA1relativ"] = (
        match_df["DA1"].astype(np.float32) - match_df["DA1"].shift(5).astype(np.float32)
    ).fillna(0)
    match_df.loc[match_df["DA1relativ"] > 10, ["DA1relativ"]] = np.float32(10.0)
    match_df["DA2relativ"] = (
        match_df["DA2"].astype(np.float32) - match_df["DA2"].shift(5).astype(np.float32)
    ).fillna(0)
    match_df.loc[match_df["DA2relativ"] > 10, ["DA2relativ"]] = np.float32(10.0)
    # Владение мячом
    match_df["Pos1_cleaned"] = match_df["Pos1"].fillna(method="ffill").fillna(0).astype(
        np.float32
    ) / np.float32(100.0)
    match_df.loc[match_df["Pos1_cleaned"] < 0.2, ["Pos1_cleaned"]] = np.float32(0.2)
    match_df.loc[match_df["Pos1_cleaned"] > 0.8, ["Pos1_cleaned"]] = np.float32(0.8)
    match_df["Pos2_cleaned"] = match_df["Pos2"].fillna(method="ffill").fillna(0).astype(
        np.float32
    ) / np.float32(100.0)
    match_df.loc[match_df["Pos2_cleaned"] < 0.2, ["Pos2_cleaned"]] = np.float32(0.2)
    match_df.loc[match_df["Pos2_cleaned"] > 0.8, ["Pos2_cleaned"]] = np.float32(0.8)
    # трансформируем удары
    match_df["Off1_norm"] = match_df["Off1"].fillna(method="ffill").fillna(0).astype(
        np.float32
    ) / np.float32(10.0)
    match_df.loc[match_df["Off1_norm"] > 1.0, ["Off1_norm"]] = np.float32(1.0)
    match_df["Off2_norm"] = match_df["Off2"].fillna(method="ffill").fillna(0).astype(
        np.float32
    ) / np.float32(10.0)
    match_df.loc[match_df["Off2_norm"] > 1.0, ["Off2_norm"]] = np.float32(1.0)
    # трансформируем удары в створ
    match_df["On1_norm"] = match_df["On1"].fillna(method="ffill").fillna(0).astype(
        np.float32
    ) / np.float32(5.0)
    match_df.loc[match_df["On1_norm"] > 1.0, ["On1_norm"]] = np.float32(1.0)
    match_df["On2_norm"] = match_df["On2"].fillna(method="ffill").fillna(0).astype(
        np.float32
    ) / np.float32(5.0)
    match_df.loc[match_df["On2_norm"] > 1.0, ["On2_norm"]] = np.float32(1.0)
    # Желтые карточки
    match_df["YC1_transformed"] = match_df["YC1"].fillna(0).astype(
        np.float32
    ) / np.float32(2.0)
    match_df.loc[match_df["YC1_transformed"] > 1.0, ["YC1_transformed"]] = np.float32(
        1.0
    )
    match_df["YC2_transformed"] = match_df["YC2"].fillna(0).astype(
        np.float32
    ) / np.float32(2.0)
    match_df.loc[match_df["YC2_transformed"] > 1.0, ["YC2_transformed"]] = np.float32(
        1.0
    )
    # трансформируем красные карточки
    match_df["RC1_transformed"] = match_df["RC1"].fillna(0).astype(np.int8)
    match_df.loc[match_df["RC1_transformed"] > 1, ["RC1_transformed"]] = np.int8(1)
    match_df["RC2_transformed"] = match_df["RC2"].fillna(0).astype(np.int8)
    match_df.loc[match_df["RC2_transformed"] > 1, ["RC2_transformed"]] = np.int8(1)
    # Замены
    match_df["Sub1_transformed"] = match_df["Sub1"].fillna(0).astype(np.int8)
    match_df.loc[match_df["Sub1_transformed"] > 1, ["Sub1_transformed"]] = np.int8(1)
    match_df["Sub2_transformed"] = match_df["Sub2"].fillna(0).astype(np.int8)
    match_df.loc[match_df["Sub2_transformed"] > 1, ["Sub2_transformed"]] = np.int8(1)
    # Угловые
    match_df["Cor1_transformed"] = match_df["Cor1"].fillna(0).astype(
        np.float32
    ) / np.float32(6.0)
    match_df.loc[match_df["Cor1_transformed"] > 1.0, ["Cor1_transformed"]] = np.float32(
        1.0
    )
    match_df["Cor2_transformed"] = match_df["Cor2"].fillna(0).astype(
        np.float32
    ) / np.float32(6.0)
    match_df.loc[match_df["Cor2_transformed"] > 1.0, ["Cor2_transformed"]] = np.float32(
        1.0
    )
    # Кэфы
    match_df["P1_transformed"] = np.log(match_df["P1"], dtype=np.float32) / 2
    match_df["P2_transformed"] = np.log(match_df["P2"], dtype=np.float32) / 2
    return match_df[match_cols].values[-1, :]


def total_probability(regression_vector1, regression_vector2):
    poisson_dict = {}
    poisson_dict[1] = {}
    poisson_dict[2] = {}
    for goal in range(7):
        poisson_dict[1][goal] = poisson.pmf(goal, regression_vector1)
        poisson_dict[2][goal] = poisson.pmf(goal, regression_vector2)

    # Считаем вероятности суммы забитых мячей
    total_matrix = np.zeros(13)
    for goal1 in range(7):
        for goal2 in range(7):
            total_matrix[goal1 + goal2] = (
                total_matrix[goal1 + goal2]
                + poisson_dict[1][goal1] * poisson_dict[2][goal2]
            )

    # Считаем вероятности забить не менее определенного количества мячей
    over_matrix = np.flip(np.cumsum(np.flip(total_matrix)))
    # Считаем вероятности забить не более определенного количества
    under_matrix = np.cumsum(total_matrix)
    return dict(
        under_05=under_matrix[0] / (under_matrix[0] + over_matrix[1]),
        over_05=over_matrix[1] / (under_matrix[0] + over_matrix[1]),
        under_10=under_matrix[0] / (under_matrix[0] + over_matrix[2]),
        over_10=over_matrix[2] / (under_matrix[0] + over_matrix[2]),
        under_15=under_matrix[1] / (under_matrix[1] + over_matrix[2]),
        over_15=over_matrix[2] / (under_matrix[1] + over_matrix[2]),
        under_20=under_matrix[1] / (under_matrix[1] + over_matrix[3]),
        over_20=over_matrix[3] / (under_matrix[1] + over_matrix[3]),
        under_25=under_matrix[2] / (under_matrix[2] + over_matrix[3]),
        over_25=over_matrix[3] / (under_matrix[2] + over_matrix[3]),
        under_30=under_matrix[2] / (under_matrix[2] + over_matrix[4]),
        over_30=over_matrix[4] / (under_matrix[2] + over_matrix[4]),
        under_35=under_matrix[3] / (under_matrix[3] + over_matrix[4]),
        over_35=over_matrix[4] / (under_matrix[3] + over_matrix[4]),
        under_40=under_matrix[3] / (under_matrix[3] + over_matrix[5]),
        over_40=over_matrix[5] / (under_matrix[3] + over_matrix[5]),
        under_45=under_matrix[4] / (under_matrix[4] + over_matrix[5]),
        over_45=over_matrix[5] / (under_matrix[4] + over_matrix[5]),
        under_50=under_matrix[4] / (under_matrix[4] + over_matrix[6]),
        over_50=over_matrix[6] / (under_matrix[4] + over_matrix[6]),
        under_55=under_matrix[5] / (under_matrix[5] + over_matrix[6]),
        over_55=over_matrix[6] / (under_matrix[5] + over_matrix[6]),
        under_60=under_matrix[5] / (under_matrix[5] + over_matrix[7]),
        over_60=over_matrix[7] / (under_matrix[5] + over_matrix[7]),
    )


def handicap_probability(regression_vector1, regression_vector2):
    poisson_dict = {}
    poisson_dict[1] = {}
    poisson_dict[2] = {}
    for goal in range(7):
        poisson_dict[1][goal] = poisson.pmf(goal, regression_vector1)
        poisson_dict[2][goal] = poisson.pmf(goal, regression_vector2)

    # Считаем вероятности суммы забитых мячей
    total_matrix = np.zeros(13)
    for goal1 in range(7):
        for goal2 in range(7):
            total_matrix[goal1 - goal2 + 6] = (
                total_matrix[goal1 - goal2 + 6]
                + poisson_dict[1][goal1] * poisson_dict[2][goal2]
            )

    # Считаем вероятности победы дома over = home_win
    hcap_home = np.cumsum(np.flip(total_matrix))
    # Считаем вероятности победы гостей under = away_win
    hcap_away = np.cumsum(total_matrix)

    return dict(
        home_m55_win=hcap_home[0] / (hcap_home[0] + hcap_away[11]),
        home_m55_lose=hcap_away[11] / (hcap_home[0] + hcap_away[11]),
        home_m50_win=hcap_home[0] / (hcap_home[0] + hcap_away[10]),
        home_m50_lose=hcap_away[10] / (hcap_home[0] + hcap_away[10]),
        home_m45_win=hcap_home[1] / (hcap_home[1] + hcap_away[10]),
        home_m45_lose=hcap_away[10] / (hcap_home[1] + hcap_away[10]),
        home_m40_win=hcap_home[1] / (hcap_home[1] + hcap_away[9]),
        home_m40_lose=hcap_away[9] / (hcap_home[1] + hcap_away[9]),
        home_m35_win=hcap_home[2] / (hcap_home[2] + hcap_away[9]),
        home_m35_lose=hcap_away[9] / (hcap_home[2] + hcap_away[9]),
        home_m30_win=hcap_home[2] / (hcap_home[2] + hcap_away[8]),
        home_m30_lose=hcap_away[2] / (hcap_home[2] + hcap_away[8]),
        home_m25_win=hcap_home[3] / (hcap_home[3] + hcap_away[8]),
        home_m25_lose=hcap_away[8] / (hcap_home[3] + hcap_away[8]),
        home_m20_win=hcap_home[3] / (hcap_home[3] + hcap_away[7]),
        home_m20_lose=hcap_away[7] / (hcap_home[3] + hcap_away[7]),
        home_m15_win=hcap_home[4] / (hcap_home[4] + hcap_away[7]),
        home_m15_lose=hcap_away[7] / (hcap_home[4] + hcap_away[7]),
        home_m10_win=hcap_home[4] / (hcap_home[4] + hcap_away[6]),
        home_m10_lose=hcap_away[6] / (hcap_home[4] + hcap_away[6]),
        home_m05_win=hcap_home[5] / (hcap_home[5] + hcap_away[6]),
        home_m05_lose=hcap_away[6] / (hcap_home[5] + hcap_away[6]),
        home_m00_win=hcap_home[5] / (hcap_home[5] + hcap_away[5]),
        home_m00_lose=hcap_away[5] / (hcap_home[5] + hcap_away[5]),
        home_p05_win=hcap_home[6] / (hcap_home[6] + hcap_away[5]),
        home_p05_lose=hcap_away[5] / (hcap_home[6] + hcap_away[5]),
        home_p10_win=hcap_home[6] / (hcap_home[6] + hcap_away[4]),
        home_p10_lose=hcap_away[4] / (hcap_home[6] + hcap_away[4]),
        home_p15_win=hcap_home[7] / (hcap_home[7] + hcap_away[4]),
        home_p15_lose=hcap_away[4] / (hcap_home[7] + hcap_away[4]),
        home_p20_win=hcap_home[7] / (hcap_home[7] + hcap_away[3]),
        home_p20_lose=hcap_away[3] / (hcap_home[7] + hcap_away[3]),
        home_p25_win=hcap_home[8] / (hcap_home[8] + hcap_away[3]),
        home_p25_lose=hcap_away[3] / (hcap_home[8] + hcap_away[3]),
        home_p30_win=hcap_home[8] / (hcap_home[8] + hcap_away[2]),
        home_p30_lose=hcap_away[2] / (hcap_home[8] + hcap_away[2]),
        home_p35_win=hcap_home[9] / (hcap_home[9] + hcap_away[2]),
        home_p35_lose=hcap_away[2] / (hcap_home[9] + hcap_away[2]),
        home_p40_win=hcap_home[9] / (hcap_home[9] + hcap_away[1]),
        home_p40_lose=hcap_away[1] / (hcap_home[9] + hcap_away[1]),
        home_p45_win=hcap_home[10] / (hcap_home[10] + hcap_away[1]),
        home_p45_lose=hcap_away[1] / (hcap_home[10] + hcap_away[1]),
        home_p50_win=hcap_home[10] / (hcap_home[10] + hcap_away[0]),
        home_p50_lose=hcap_away[0] / (hcap_home[10] + hcap_away[0]),
        home_p55_win=hcap_home[11] / (hcap_home[11] + hcap_away[0]),
        home_p55_lose=hcap_away[0] / (hcap_home[11] + hcap_away[0]),
    )


def make_predict():
    """
    :param:
    :return:
    """
    output_dict = {}
    data_dir = os.path.join(os.path.abspath("../../"), "data", "*.csv")
    match_cols: list = features_list()
    for file_path in glob(data_dir):
        file_num = os.path.basename(file_path).split(".")[0]
        output_dict[file_num] = {}
        input_vector = create_predict_vector(file_path, match_cols)
        (
            output_dict[file_num]["mc_home"],
            output_dict[file_num]["mc_draw"],
            output_dict[file_num]["mc_away"],
        ) = np.flip(
            MODEL_DICT["1x2"]["1"].predict(input_vector, prediction_type="Probability")
        )
        output_dict[file_num].update(
            total_probability(
                MODEL_DICT["total"]["1"].predict(input_vector) * 21,
                MODEL_DICT["total"]["2"].predict(input_vector) * 21,
            )
        )
        output_dict[file_num].update(
            handicap_probability(
                MODEL_DICT["handicap"]["1"].predict(input_vector) * 21,
                MODEL_DICT["handicap"]["2"].predict(input_vector) * 21,
            )
        )
    pd.DataFrame.from_dict(output_dict, orient="index").to_csv("./output.csv")


def console_predict():
    yaml_path = os.path.join(os.path.abspath("./models"), "model_path.yml")
    with open(yaml_path, "r") as yml:
        model_path_dict = yaml.load(yml, Loader=yaml.SafeLoader)
    for model_key, model_path in model_path_dict.items():
        model_name, model_num = model_key.split(":")
        if model_name in MODEL_DICT:
            MODEL_DICT[model_name][model_num] = CatBoost()
        else:
            MODEL_DICT[model_name] = {}
            MODEL_DICT[model_name][model_num] = CatBoost()
        MODEL_DICT[model_name][model_num].load_model(model_path)
    make_predict()


def calculate_predict(
    model_fold_dict: dict, input_vector: np.array, prediction_type: str
):
    return sum(
        model_fold_dict[fold].predict(input_vector, prediction_type=prediction_type)
        for fold in model_fold_dict.keys()
    ) / len(model_fold_dict.keys())


def console_folded_predict(
    models_dict: dict, model_feature_dict: dict, flip_multiclass_output: bool
):
    """
    :return:
    1. Make dict of models for all types and folds via yaml
    2. Make features list
    """
    output_dict = {}
    data_dir = os.path.join(os.path.abspath("../../"), "data", "*.csv")
    match_cols: list = features_list()
    for file_path in glob(data_dir):
        file_num = os.path.basename(file_path).split(".")[0]
        output_dict[file_num] = {}
        input_vector = create_predict_vector(file_path, match_cols)
        if flip_multiclass_output:
            (
                output_dict[file_num]["mc_home"],
                output_dict[file_num]["mc_draw"],
                output_dict[file_num]["mc_away"],
            ) = np.flip(
                calculate_predict(
                    models_dict["multiclass"],
                    input_vector[np.array(model_feature_dict["multiclass"])],
                    "Probability",
                )
            )
        else:
            (
                output_dict[file_num]["mc_home"],
                output_dict[file_num]["mc_draw"],
                output_dict[file_num]["mc_away"],
            ) = calculate_predict(
                models_dict["multiclass"],
                input_vector[np.array(model_feature_dict["multiclass"])],
                "Probability",
            )
        output_dict[file_num].update(
            total_probability(
                calculate_predict(
                    models_dict["total_over"],
                    input_vector[np.array(model_feature_dict["total_over"])],
                    "RawFormulaVal",
                )
                * 21,
                calculate_predict(
                    models_dict["total_under"],
                    input_vector[np.array(model_feature_dict["total_under"])],
                    "RawFormulaVal",
                )
                * 21,
            )
        )
        output_dict[file_num].update(
            handicap_probability(
                calculate_predict(
                    models_dict["handicap_home"],
                    input_vector[np.array(model_feature_dict["handicap_home"])],
                    "RawFormulaVal",
                )
                * 21,
                calculate_predict(
                    models_dict["handicap_home"],
                    input_vector[np.array(model_feature_dict["handicap_home"])],
                    "RawFormulaVal",
                )
                * 21,
            )
        )
    pd.DataFrame.from_dict(output_dict, orient="index").to_csv("./output.csv")
