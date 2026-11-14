"""Load IEEE-CIS data and split it by time.

The split is chronological on ``TransactionDT`` (seconds from an unknown reference
point). A random split would let the model learn from transactions that happen
after the ones it is evaluated on, which inflates every metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "isFraud"
TIME_COL = "TransactionDT"
ID_COL = "TransactionID"


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, df in (("train", self.train), ("val", self.val), ("test", self.test)):
            rows.append(
                {
                    "split": name,
                    "rows": len(df),
                    "fraud_rate": round(float(df[TARGET].mean()), 4),
                    "dt_min": int(df[TIME_COL].min()),
                    "dt_max": int(df[TIME_COL].max()),
                }
            )
        return pd.DataFrame(rows)


def load_ieee(data_dir: str | Path = "data/raw", nrows: int | None = None) -> pd.DataFrame:
    """Load train_transaction.csv and left-join train_identity.csv on TransactionID."""
    data_dir = Path(data_dir)
    txn_path = data_dir / "train_transaction.csv"
    if not txn_path.exists():
        raise FileNotFoundError(
            f"{txn_path} not found. Run `make data` first (see scripts/download_data.sh)."
        )
    txn = pd.read_csv(txn_path, nrows=nrows)
    id_path = data_dir / "train_identity.csv"
    if id_path.exists():
        ident = pd.read_csv(id_path)
        txn = txn.merge(ident, on=ID_COL, how="left")
    return reduce_memory(txn)


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns; IEEE-CIS is ~2 GB in memory without this."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def time_split(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15) -> Split:
    """Chronological train / val / test split on TransactionDT."""
    if not 0 < val_frac < 1 or not 0 < test_frac < 1 or val_frac + test_frac >= 1:
        raise ValueError("val_frac and test_frac must be in (0, 1) and sum to less than 1")
    ordered = df.sort_values(TIME_COL, kind="mergesort").reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * (1 - val_frac - test_frac))
    val_end = int(n * (1 - test_frac))
    return Split(
        train=ordered.iloc[:train_end].copy(),
        val=ordered.iloc[train_end:val_end].copy(),
        test=ordered.iloc[val_end:].copy(),
    )


def numeric_features(df: pd.DataFrame) -> list[str]:
    """Numeric columns usable by the baseline (excludes target, ID and raw time)."""
    excluded = {TARGET, ID_COL, TIME_COL}
    return [c for c in df.select_dtypes(include="number").columns if c not in excluded]


def make_synthetic(n: int = 5000, fraud_rate: float = 0.035, seed: int = 0) -> pd.DataFrame:
    """Small IEEE-CIS-shaped dataset for tests and smoke runs without Kaggle access.

    Fraud rows get larger amounts, more night-time activity and higher card velocity,
    so a model has a real (but imperfect) signal to learn.
    """
    rng = np.random.default_rng(seed)
    is_fraud = (rng.random(n) < fraud_rate).astype(int)
    dt = np.sort(rng.integers(86_400, 86_400 * 180, size=n))
    amount = np.where(is_fraud, rng.lognormal(5.0, 1.0, n), rng.lognormal(4.0, 1.0, n))
    hour = (dt // 3600) % 24
    night = ((hour < 6) & (rng.random(n) < np.where(is_fraud, 0.6, 0.15))).astype(int)
    velocity = rng.poisson(np.where(is_fraud, 4.0, 1.5))
    df = pd.DataFrame(
        {
            ID_COL: np.arange(2_987_000, 2_987_000 + n),
            TIME_COL: dt,
            "TransactionAmt": amount.round(2),
            "ProductCD": rng.choice(list("WCRHS"), n, p=[0.74, 0.12, 0.06, 0.05, 0.03]),
            "card4": rng.choice(["visa", "mastercard", "discover", "american express"], n),
            "card1": rng.integers(1000, 18000, n),
            "C1": velocity + rng.poisson(1.0, n),
            "D1": np.where(rng.random(n) < 0.1, np.nan, rng.integers(0, 600, n)),
            "night": night,
            TARGET: is_fraud,
        }
    )
    return df
