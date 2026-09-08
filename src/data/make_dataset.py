"""Construction du dataset baseline : nettoyage texte + split stratifie.

Entree  : data/x_train.csv + data/y_train.csv
Sortie  : data/processed/{train,val}.parquet (colonnes text_combined, prdtypecode, label)
"""

import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import load_params, resolve
from src.data.download_data import check_raw_data
from src.data.text_cleaning import build_text_combined

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def build_dataframe(x_path, y_path) -> pd.DataFrame:
    x = pd.read_csv(x_path, index_col=0)
    y = pd.read_csv(y_path, index_col=0)
    df = x.join(y)
    if df["prdtypecode"].isna().any():
        raise ValueError("Jointure x/y incomplete : labels manquants")

    log.info("Donnees brutes : %d lignes", len(df))
    df["text_combined"] = [
        build_text_combined(d, desc)
        for d, desc in zip(df["designation"], df["description"])
    ]
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["prdtypecode"])
    return df[["productid", "imageid", "text_combined", "prdtypecode", "label"]]


def split_and_save(df: pd.DataFrame) -> None:
    params = load_params()
    out_dir = resolve(params["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df = train_test_split(
        df,
        test_size=params["split"]["val_size"],
        stratify=df["label"],
        random_state=params["split"]["random_state"],
    )
    train_df.to_parquet(out_dir / "train.parquet", index=False)
    val_df.to_parquet(out_dir / "val.parquet", index=False)
    log.info("Train : %d lignes -> %s", len(train_df), out_dir / "train.parquet")
    log.info("Val   : %d lignes -> %s", len(val_df), out_dir / "val.parquet")


def main() -> None:
    if not check_raw_data():
        raise SystemExit(1)
    params = load_params()["data"]
    df = build_dataframe(resolve(params["raw_x"]), resolve(params["raw_y"]))
    split_and_save(df)


if __name__ == "__main__":
    main()
