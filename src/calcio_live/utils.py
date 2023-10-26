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
model_dict = {}

data_types_dict = {
    "Id": np.int32,
    "StatTime": str,
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

cols = [
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
]

backup_cols = [
    "Id",
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
    "TimeSnapshot",
    "D",
    "I",
    "Active.1",
    "Time",
    "Minute.1",
    "RawTime",
    "Score1.1",
    "Score2.1",
    "Period.1",
    "Periods",
    "Serve",
    "W1",
    "WX",
    "W2",
    "X1",
    "X2",
    "W12",
    "TotalValue",
    "Over",
    "Under",
    "Hand1Value",
    "H1",
    "H2",
]

usecols = [
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

match_cols = [
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
    print(os.path.join(os.path.abspath("./"), "model_path.yml"))
    token_path = os.path.realpath("../credential.txt")
    with open(token_path, "r") as container:
        for line in container:
            if frmwork in line:
                login, psw = line.split(" ")[1], line.split(" ")[2].split("\n")[0]
                return login, psw


def neptune_download(model_name, model_num, path_to_model):
    """
    :param model_num:
    :param model_type:
    """
    _, api_key = get_credential()
    # print(model_name + '-' + str(model_num))
    # print(neptune)
    model_version = neptune.init_model_version(
        project="scomesse/football",
        model=model_name,
        api_token=api_key,
        with_id=model_name + "-" + str(model_num),
    )

    try:
        # print(f"Загружаем модель {model_name} n.{model_num}")
        model_version["model"].download(path_to_model)
        model_version.stop()
        logging.info(f"Downloaded: {path_to_model}")
    except Exception:
        logging.error(f"Downloaded neptune: {path_to_model}")


def load_model(model_type_order: str, model_name: str, num: str):
    """
    :param model_type_order: '1x2:1', 'total:1', 'total:2', 'handicap:1', 'handicap:2'
    :param model_name: FOOT-LIVEMC
    :param num: 3
    :return: local model path
    """
    model_type, order = model_type_order.split(":")
    model_num = str(num)
    models_dir = os.path.join(INPUT_DIR, "models")
    PATH_TO_MODEL = os.path.join(
        models_dir, f"booster_{model_type}_{model_name}_{model_num}.model"
    )
    neptune_download(model_name, model_num, PATH_TO_MODEL)
    if model_type in model_dict:
        model_dict[model_type].update({order: CatBoost()})
    else:
        model_dict[model_type] = {order: CatBoost()}
    model_dict[model_type][order].load_model(PATH_TO_MODEL)
    return PATH_TO_MODEL


def create_predict_vector(file_path: str, p1pxp2_align: bool):
    match_df = pd.read_csv(
        file_path, sep=";", index_col=False, names=cols, skiprows=1, usecols=usecols
    )
    match_df.iloc[0, :] = match_df.iloc[0, :].fillna(0)
    match_df = match_df.ffill()
    P1, PX, P2 = pd.read_csv(
        file_path,
        sep=";",
        nrows=1,
        header=None,
        dtype={0: np.float32, 1: np.float32, 2: np.float32},
    ).values[0]
    if p1pxp2_align:
        if (P1 > 0) & (PX > 0) & (P2 > 0):
            psum = 1 / P1 + 1 / PX + 1 / P2
        else:
            psum = 1
    else:
        psum = 1
    match_df[["P1", "P2"]] = P1 * psum, P2 * psum
    match_df["min_norm"] = match_df["Minute"].astype(np.float32) / 50
    # трансформируем голы
    match_df[match_df["Score1"].isna()] = 0
    match_df["Score1_norm"] = (
        match_df["Score1"].ffill().fillna(0).astype(np.float32) / 4
    )
    match_df.loc[match_df["Score1"] > 3, ["Score1_norm"]] = 1.0
    match_df[match_df["Score2"].isna()] = 0
    match_df["Score2_norm"] = (
        match_df["Score2"].ffill().fillna(0).astype(np.float32) / 4
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
        60 + (match_df["A1"].astype(np.float32) - 60) / 4
    ) / 75
    match_df["A2_scaled"] = match_df["A2"].astype(np.float32) / 75
    match_df.loc[match_df["A2"] >= 60, ["A2_scaled"]] = (
        60 + (match_df["A2"].astype(np.float32) - 60) / 4
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
    match_df["Pos1_cleaned"] = match_df["Pos1"].ffill().fillna(0).astype(
        np.float32
    ) / np.float32(100.0)
    match_df.loc[match_df["Pos1_cleaned"] < 0.2, ["Pos1_cleaned"]] = np.float32(0.2)
    match_df.loc[match_df["Pos1_cleaned"] > 0.8, ["Pos1_cleaned"]] = np.float32(0.8)
    match_df["Pos2_cleaned"] = match_df["Pos2"].ffill().fillna(0).astype(
        np.float32
    ) / np.float32(100.0)
    match_df.loc[match_df["Pos2_cleaned"] < 0.2, ["Pos2_cleaned"]] = np.float32(0.2)
    match_df.loc[match_df["Pos2_cleaned"] > 0.8, ["Pos2_cleaned"]] = np.float32(0.8)
    # трансформируем удары
    match_df["Off1_norm"] = match_df["Off1"].ffill().fillna(0).astype(
        np.float32
    ) / np.float32(10.0)
    match_df.loc[match_df["Off1_norm"] > 1.0, ["Off1_norm"]] = np.float32(1.0)
    match_df["Off2_norm"] = match_df["Off2"].ffill().fillna(0).astype(
        np.float32
    ) / np.float32(10.0)
    match_df.loc[match_df["Off2_norm"] > 1.0, ["Off2_norm"]] = np.float32(1.0)
    # трансформируем удары в створ
    match_df["On1_norm"] = match_df["On1"].ffill().fillna(0).astype(
        np.float32
    ) / np.float32(5.0)
    match_df.loc[match_df["On1_norm"] > 1.0, ["On1_norm"]] = np.float32(1.0)
    match_df["On2_norm"] = match_df["On2"].ffill().fillna(0).astype(
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
    return match_df


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
    :param file_path:
    :return:
    """
    output_dict = {}
    data_dir = os.path.join(os.path.abspath("../../"), "data", "*.csv")
    for file_path in glob(data_dir):
        file_num = os.path.basename(file_path).split(".")[0]
        output_dict[file_num] = {}
        input_vector = create_predict_vector(file_path)[match_cols].values[-1, :]
        (
            output_dict[file_num]["mc_home"],
            output_dict[file_num]["mc_draw"],
            output_dict[file_num]["mc_away"],
        ) = np.flip(
            model_dict["1x2"]["1"].predict(input_vector, prediction_type="Probability")
        )
        output_dict[file_num].update(
            total_probability(
                model_dict["total"]["1"].predict(input_vector) * 21,
                model_dict["total"]["2"].predict(input_vector) * 21,
            )
        )
        output_dict[file_num].update(
            handicap_probability(
                model_dict["handicap"]["1"].predict(input_vector) * 21,
                model_dict["handicap"]["2"].predict(input_vector) * 21,
            )
        )
    pd.DataFrame.from_dict(output_dict, orient="index").to_csv("./output.csv")


def console_predict():
    # print(os.path.join(os.path.abspath('./models'), 'model_path.yml'))
    # return {}
    yaml_path = os.path.join(os.path.abspath("./models"), "model_path.yml")
    with open(yaml_path, "r") as yml:
        model_path_dict = yaml.load(yml, Loader=yaml.SafeLoader)
    for model_key, model_path in model_path_dict.items():
        model_name, model_num = model_key.split(":")
        if model_name in model_dict:
            model_dict[model_name][model_num] = CatBoost()
        else:
            model_dict[model_name] = {}
            model_dict[model_name][model_num] = CatBoost()
        model_dict[model_name][model_num].load_model(model_path)
    make_predict()


def predict_by_model_type(preload_models_dict: dict, input_vector: np.array):
    if "home" in preload_models_dict:
        preds_dict = {}
        for team, preload_models_dict_team in preload_models_dict.items():
            preds_dict[team] = sum(
                preloaded_model.predict(input_vector) * 21
                for _, preloaded_model in preload_models_dict_team.items()
            ) / len(preload_models_dict_team)
        return preds_dict
    else:
        preds = sum(
            preloaded_model.predict(input_vector, prediction_type="Probability")
            for _, preloaded_model in preload_models_dict.items()
        ) / len(preload_models_dict)
        return preds


def load_model_info_from_yaml() -> dict:
    yaml_path = os.path.join(os.path.split(INPUT_DIR)[0], "models_info.yaml")
    try:
        with open(yaml_path, "r") as yml:
            yaml_dict = yaml.load(yml, Loader=yaml.SafeLoader)
            return yaml_dict
    except:
        raise Exception("No models setup file found")


def load_model_to_dict(model_type: str, model_dict: dict) -> dict:
    model_path = model_dict["path"]
    folds_quantity = model_dict["fold_quantity"]
    if model_type in ["FOOT-LIVETOTAL", "FOOT-LIVEHCAP"]:
        preds_dict = {"home": {}, "away": {}}
        preds_dict["home"] = {
            kfold_num: CatBoost().load_model(
                os.path.join(model_path, f"booster_reg1_{kfold_num}.model")
            )
            for kfold_num in range(folds_quantity)
        }
        preds_dict["away"] = {
            kfold_num: CatBoost().load_model(
                os.path.join(model_path, f"booster_reg2_{kfold_num}.model")
            )
            for kfold_num in range(folds_quantity)
        }
        return preds_dict
    else:
        return {
            kfold_num: CatBoost().load_model(
                os.path.join(model_path, f"booster_{kfold_num}.model")
            )
            for kfold_num in range(folds_quantity)
        }


def console_predict_v2(model_type_list: list):
    model_dict = load_model_info_from_yaml()
    preload_models_dict = {}
    for model_type in model_dict:
        if model_type in model_type_list:
            preload_models_dict[model_type] = load_model_to_dict(
                model_type, model_dict[model_type]
            )
    output_dict = {}
    data_dir = os.path.join(
        os.path.split(os.path.split(INPUT_DIR)[0])[0], "data", "*.csv"
    )
    for file_path in glob(data_dir):
        file_num = os.path.basename(file_path).split(".")[0]
        output_dict[file_num] = {}
        for model_type in model_type_list:
            if "p1pxp2_align" in model_dict[model_type]:
                p1pxp2_align = model_dict[model_type]["p1pxp2_align"]
            else:
                p1pxp2_align = False
            if model_type == "FOOT-LIVEMC":
                input_vector = create_predict_vector(file_path, p1pxp2_align)
                (
                    output_dict[file_num]["mc_home"],
                    output_dict[file_num]["mc_draw"],
                    output_dict[file_num]["mc_away"],
                ) = predict_by_model_type(
                    preload_models_dict[model_type],
                    input_vector[
                        yaml.load(
                            model_dict[model_type]["selected_columns"],
                            Loader=yaml.SafeLoader,
                        )
                    ].values[-1, :],
                )
            elif model_type == "FOOT-LIVETOTAL":
                input_vector = create_predict_vector(file_path, p1pxp2_align)
                preds_dict = predict_by_model_type(
                    preload_models_dict[model_type],
                    input_vector[
                        yaml.load(
                            model_dict[model_type]["selected_columns"],
                            Loader=yaml.SafeLoader,
                        )
                    ].values[-1, :],
                )
                output_dict[file_num].update(
                    total_probability(preds_dict["home"], preds_dict["away"])
                )
            elif model_type == "FOOT-LIVEHCAP":
                input_vector = create_predict_vector(file_path, p1pxp2_align)
                preds_dict = predict_by_model_type(
                    preload_models_dict[model_type],
                    input_vector[
                        yaml.load(
                            model_dict[model_type]["selected_columns"],
                            Loader=yaml.SafeLoader,
                        )
                    ].values[-1, :],
                )
                output_dict[file_num].update(
                    handicap_probability(preds_dict["home"], preds_dict["away"])
                )
    pd.DataFrame.from_dict(output_dict, orient="index").to_csv(
        os.path.abspath("../../output.csv")
    )


def add_model_info_to_yaml(model_type: str, model_description_dict: dict) -> bool:
    yaml_path = os.path.join(os.path.split(INPUT_DIR)[0], "models_info.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as yml:
            yaml_dict = yaml.load(yml, Loader=yaml.SafeLoader)
        if not (isinstance(yaml_dict, dict)):
            yaml_dict = {}
        yaml_dict[model_type] = model_description_dict
    else:
        yaml_dict = {model_type: model_description_dict}
    with open(yaml_path, "w") as yml:
        yaml.dump(yaml_dict, yml, default_flow_style=False)

    return True


def delete_files_in_directory(directory_path: str) -> bool:
    try:
        files = os.listdir(directory_path)
        for file in files:
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        return True
        # print("All files deleted successfully.")
    except OSError:
        return False
        # print("Error occurred while deleting files.")


def check_dir_and_makedir(existed_dir: str, new_dir_name: str) -> str:
    check_path = os.path.join(existed_dir, new_dir_name)
    if not os.path.exists(check_path):
        os.makedirs(check_path)
    return check_path


def get_fold_quantity_and_download(model_type: str, model_num: str) -> bool:
    _, api_key = get_credential()
    model_version_parameters = dict(
        project="scomesse/football",
        model=model_type,
        api_token=api_key,
        with_id=model_type + "-" + str(model_num),
    )
    model_version = neptune.init_model_version(**model_version_parameters)
    model_version_structure = model_version.get_structure()
    if "description" in model_version_structure:
        model_description_dict = {
            key: model_version["description/" + key].fetch()
            for key, value in model_version_structure["description"].items()
        }
        if "kfold_splits" in model_version_structure["description"]:
            folds_quantity = model_version["description/kfold_splits"].fetch()
        elif "fold_quantity" in model_version_structure["description"]:
            folds_quantity = model_version["description/fold_quantity"].fetch()
        elif "model_0_description" in model_version_structure["models"]:
            folds_quantity = model_version[
                "models/model_0_description/kfold_splits"
            ].fetch()
        model_description_dict["fold_quantity"] = folds_quantity
    else:
        print("Folds_quantity not found")
        folds_quantity = 0
    print("folds_quantity = ", folds_quantity)
    main_models_dir = check_dir_and_makedir(INPUT_DIR, "models")
    model_description_dict["path"] = check_dir_and_makedir(main_models_dir, model_type)
    _ = delete_files_in_directory(model_description_dict["path"])
    model_description_dict["version"] = model_num

    if "models" in model_version_structure:
        for model_name in model_version_structure["models"].keys():
            final_name = model_name.split(".")[0].replace("model", "booster")
            model_version[f"models/{model_name}"].download(
                os.path.join(model_description_dict["path"], f"{final_name}.model")
            )
    else:
        print(f"no models in this version")
    model_version.stop()
    _ = add_model_info_to_yaml(model_type, model_description_dict)
    return True


def download_folded_models(**model_type_dict):
    for model_type, options_dict in model_type_dict.items():
        if options_dict["version"] is not None:
            model_num = str(options_dict["version"])
            get_fold_quantity_and_download(model_type, model_num)
