"""Feature engineering for FraudShield.

Two kinds of features, built so the same logic can run on a live stream later:

1. **Row features** use only the transaction itself (time of day, amount shape,
   time-relative D columns, a client id).
2. **History features** use only *earlier* transactions of the same card or client
   (counts and spend in the last hour / day, gap since the previous one, how this
   amount compares with the past average). No labels are used, and nothing from the
   future leaks in. In week 3 these move to Redis, keyed the same way.

Categorical columns are mapped to the categories seen in training; anything new at
validation, test or serving time becomes missing, which XGBoost handles natively.

Why the D columns are rewritten: many of them count days since some event (card
opened, last seen...). Their raw values keep growing as time passes, so the ranges
a model learns in training no longer match a month later; that drift is why the
baseline dropped from 0.39 PR-AUC on validation to 0.17 on test. ``day - Dn`` turns
"days since the card was opened" into "the day the card was opened", which stays
constant for a card no matter when we look at it.
"""

from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ml.src.data import ID_COL, TARGET, TIME_COL

SECONDS_PER_DAY = 86_400

# "Days since X" columns that grow with time; replaced by day - Dn.
DRIFTING_D_COLS = ["D1", "D2", "D4", "D10", "D11", "D15"]

# History windows (seconds) and keys.
WINDOWS = {"1h": 3_600, "24h": SECONDS_PER_DAY}
HISTORY_KEYS = ["card1", "uid"]

# Raw columns never used as model inputs.
DROP_COLS = [ID_COL, TIME_COL, TARGET, "day", "uid", *DRIFTING_D_COLS]


def _quiet_fragmentation(fn):
    """Adding columns one by one to a 400-column frame makes pandas warn about
    fragmentation; in place it saves a full copy of the data, so the warning is expected."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            return fn(*args, **kwargs)

    return wrapper


@_quiet_fragmentation
def add_row_features(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """Features from the transaction alone. ``inplace=True`` saves a full copy of the data."""
    out = df if inplace else df.copy()
    t = out[TIME_COL]
    out["day"] = t // SECONDS_PER_DAY
    out["hour"] = (t // 3_600) % 24
    out["dow"] = out["day"] % 7

    amt = out["TransactionAmt"].astype("float64")
    out["amt_log"] = np.log1p(amt)
    # Fractional cents: converted foreign-currency amounts rarely end in .00
    out["amt_cents"] = (amt - np.floor(amt)).round(3)
    out["amt_is_round"] = (out["amt_cents"] == 0).astype("int8")

    for col in DRIFTING_D_COLS:
        if col in out:
            out[f"{col}_start"] = out["day"] - out[col]

    # Approximate client id: card + billing zip + the day the card was opened.
    out["uid"] = (
        out["card1"].astype("string").fillna("na")
        + "_"
        + out.get("addr1", pd.Series(np.nan, index=out.index)).astype("string").fillna("na")
        + "_"
        + out["D1_start"].astype("string").fillna("na")
    )
    return out


def _window_stats(times: np.ndarray, codes: np.ndarray, amounts: np.ndarray, window: int):
    """Count and spend of *earlier* transactions with the same code within `window` seconds.

    Vectorised: sort by (code, time), then binary-search each row's window start inside
    its own group. Ties at the same second count as earlier, which is what a stream sees.
    """
    order = np.lexsort((times, codes))
    t = times[order].astype("int64")
    c = codes[order].astype("int64")
    a = amounts[order]
    span = int(t.max() - t.min()) + window + 1
    key = c * span + (t - t.min())
    left = np.searchsorted(key, key - window, side="left")
    idx = np.arange(len(key))
    count = idx - left
    csum = np.concatenate([[0.0], np.cumsum(a)])
    spend = csum[idx] - csum[left]
    out_count = np.empty_like(count)
    out_spend = np.empty_like(spend)
    out_count[order] = count
    out_spend[order] = spend
    return out_count, out_spend


def history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Past-only behaviour features per card and per client, as a new frame on df's index.

    Expects add_row_features first. Run it on the full time-ordered history (train + val +
    test together): each row only ever looks backwards, exactly like a live stream that has
    seen everything before it. Works on a small slice of columns, so memory stays low.
    """
    s = df[[TIME_COL, "TransactionAmt", *HISTORY_KEYS]].sort_values(TIME_COL, kind="mergesort")
    times = s[TIME_COL].to_numpy()
    amt = s["TransactionAmt"].astype("float64")
    out = pd.DataFrame(index=s.index)

    for key in HISTORY_KEYS:
        codes = pd.factorize(s[key])[0]  # NaN key -> -1, its own group
        g_time = s[TIME_COL].groupby(codes, sort=False)
        g_amt = amt.groupby(codes, sort=False)

        prior_n = g_amt.cumcount().astype("float64")
        prior_sum = g_amt.cumsum() - amt
        prior_mean = (prior_sum / prior_n).where(prior_n > 0)
        out[f"{key}_n_prior"] = prior_n
        out[f"{key}_amt_vs_mean"] = amt / prior_mean
        out[f"{key}_secs_since_prev"] = s[TIME_COL] - g_time.shift(1)

        for name, seconds in WINDOWS.items():
            cnt, spend = _window_stats(times, codes, amt.to_numpy(), seconds)
            out[f"{key}_n_{name}"] = cnt
            out[f"{key}_spend_{name}"] = spend

        if key == "card1":
            # Distinct clients seen on this card so far: many = card number shared/tested.
            first_seen = ~s.duplicated(["card1", "uid"])
            out["card1_n_uids"] = first_seen.astype("int32").groupby(codes, sort=False).cumsum()

    return out.astype("float32").reindex(df.index)


@_quiet_fragmentation
def add_history_features(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """Adds history_features() columns to df (a copy unless ``inplace=True``)."""
    out = df if inplace else df.copy()
    hist = history_features(out)
    for col in hist.columns:
        out[col] = hist[col]
    return out


def _is_categorical(s: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


@dataclass
class FeatureBuilder:
    """Fit on training rows; transform any rows (validation, test, serving) the same way."""

    categories: dict[str, list[str]] = field(default_factory=dict)
    columns: list[str] = field(default_factory=list)

    def fit(self, train: pd.DataFrame) -> FeatureBuilder:
        feats = train.drop(columns=[c for c in DROP_COLS if c in train])
        self.columns = list(feats.columns)
        self.categories = {
            c: sorted(feats[c].dropna().astype(str).unique().tolist())
            for c in self.columns
            if _is_categorical(feats[c])
        }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.columns:
            raise RuntimeError("FeatureBuilder.fit() must be called before transform()")
        x = df.reindex(columns=self.columns)
        for c, cats in self.categories.items():
            x[c] = pd.Categorical(x[c].astype("string"), categories=cats)
        numeric = [c for c in self.columns if c not in self.categories]
        x[numeric] = x[numeric].astype("float32")
        return x

    @property
    def categorical_columns(self) -> list[str]:
        return list(self.categories)


def build_features(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """Row + history features for a raw IEEE-CIS frame (all rows, any order).

    Use ``inplace=True`` on the full dataset: it adds columns to df instead of copying it.
    """
    out = add_row_features(df, inplace=inplace)
    return add_history_features(out, inplace=True)
