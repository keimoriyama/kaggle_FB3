from utils import seed_everything, get_logger, get_score
from config import CFG
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import pandas as pd
from train import train_loop
import mlflow

CFG.output_dir = "."
CFG.logger = get_logger()

import os


def check_kaggle_env():
    env_set = set(os.environ.keys())

    return "KAGGLE_URL_BASE" in env_set


def main():
    seed_everything(seed=42)
    # ====================================================
    # Data Loading
    # ====================================================
    base_path = ""
    if check_kaggle_env():
        base_path = "../../../input/feedback-prize-english-language-learning/"
    else:
        base_path = "../input/feedback-prize-english-language-learning/"
    train = pd.read_csv(base_path + "train.csv")
    test = pd.read_csv(base_path + "test.csv")
    submission = pd.read_csv(base_path + "sample_submission.csv")

    print(f"train.shape: {train.shape}")
    print(f"test.shape: {test.shape}")
    print(f"submission.shape: {submission.shape}")

    # ====================================================
    # CV split
    # ====================================================
    Fold = MultilabelStratifiedKFold(
        n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed
    )
    for n, (train_index, val_index) in enumerate(
        Fold.split(train, train[CFG.target_cols])
    ):
        train.loc[val_index, "fold"] = int(n)
    train["fold"] = train["fold"].astype(int)
    print(train.groupby("fold").size())

    if CFG.debug:
        train = train.sample(n=1000, random_state=0).reset_index(drop=True)

    def get_result(oof_df):
        labels = oof_df[CFG.target_cols].values
        preds = oof_df[[f"pred_{c}" for c in CFG.target_cols]].values
        score, scores = get_score(labels, preds)
        CFG.logger.info(f"Score: {score:<.4f}  Scores: {scores}")

    if CFG.train:
        oof_df = pd.DataFrame()
        if CFG.mlflow:
            mlflow.set_experiment(CFG.experiment)
            tracking_uri = mlflow.get_tracking_uri()
            print("Current tracking uri: {}".format(tracking_uri))
            mlflow.start_run()
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                _oof_df = train_loop(train, fold)
                oof_df = pd.concat([oof_df, _oof_df])
                CFG.logger.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        oof_df = oof_df.reset_index(drop=True)
        CFG.logger.info(f"========== CV ==========")
        get_result(oof_df)
        oof_df.to_pickle(CFG.output_dir + "oof_df.pkl")
        if CFG.mlflow:
            mlflow.end_run()


if __name__ == "__main__":
    main()
