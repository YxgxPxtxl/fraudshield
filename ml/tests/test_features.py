import numpy as np
import pandas as pd
import pytest

from ml.src.data import TARGET, TIME_COL, make_synthetic, time_split
from ml.src.features import (
    FeatureBuilder,
    add_history_features,
    add_row_features,
    build_features,
)


def _txns(rows):
    """Tiny frame: (TransactionDT, card1, amount)."""
    df = pd.DataFrame(rows, columns=[TIME_COL, "card1", "TransactionAmt"])
    df["addr1"] = 100.0
    df["D1"] = 0.0
    df["TransactionID"] = range(len(df))
    df[TARGET] = 0
    return df


def test_row_features_time_and_amount():
    df = _txns([(86_400 * 3 + 3_600 * 5, 1, 12.5), (86_400 * 3 + 3_600 * 23, 1, 40.0)])
    f = add_row_features(df)
    assert f["day"].tolist() == [3, 3]
    assert f["hour"].tolist() == [5, 23]
    assert f["amt_cents"].tolist() == [0.5, 0.0]
    assert f["amt_is_round"].tolist() == [0, 1]


def test_drifting_d_column_becomes_constant_start_day():
    # Same card seen 10 days apart: D1 grows by 10, but the start day stays the same.
    df = _txns([(86_400 * 20, 1, 10.0), (86_400 * 30, 1, 10.0)])
    df["D1"] = [5.0, 15.0]
    f = add_row_features(df)
    assert f["D1_start"].tolist() == [15, 15]
    assert f["uid"].nunique() == 1


def test_history_counts_only_earlier_transactions():
    df = _txns(
        [
            (1_000, 7, 10.0),
            (1_000 + 600, 7, 20.0),  # 10 min later
            (1_000 + 3_000, 7, 30.0),  # 50 min after first
            (1_000 + 7_200, 7, 40.0),  # 2 h after first
            (1_000 + 100, 8, 99.0),  # another card
        ]
    )
    f = build_features(df).set_index("TransactionID")
    assert f.loc[0, "card1_n_1h"] == 0  # first ever: no history
    assert f.loc[1, "card1_n_1h"] == 1
    assert f.loc[2, "card1_n_1h"] == 2
    assert f.loc[2, "card1_spend_1h"] == pytest.approx(30.0)
    assert f.loc[3, "card1_n_1h"] == 0  # the others are > 1 h earlier
    assert f.loc[3, "card1_n_24h"] == 3
    assert f.loc[4, "card1_n_24h"] == 0  # card 8 has its own history
    assert f.loc[1, "card1_secs_since_prev"] == 600
    assert f.loc[3, "card1_amt_vs_mean"] == pytest.approx(40.0 / 20.0)


def test_history_is_unaffected_by_future_rows():
    df = make_synthetic(3000, seed=2)
    df["addr1"] = 300.0
    full = build_features(df)
    cutoff = df[TIME_COL].quantile(0.6)
    past_only = build_features(df[df[TIME_COL] <= cutoff])
    cols = [c for c in full.columns if c.startswith(("card1_", "uid_"))]
    pd.testing.assert_frame_equal(full.loc[past_only.index, cols], past_only[cols])


def test_window_counts_match_brute_force():
    rng = np.random.default_rng(0)
    df = _txns(
        [
            (int(t), int(c), 1.0)
            for t, c in zip(rng.integers(0, 20_000, 400), rng.integers(0, 5, 400), strict=True)
        ]
    )
    f = add_history_features(add_row_features(df))
    t, c = df[TIME_COL].to_numpy(), df["card1"].to_numpy()
    for i in rng.choice(len(df), 50, replace=False):
        same = c == c[i]
        earlier = (t < t[i]) | ((t == t[i]) & (np.arange(len(df)) < i))
        # ties at the same second: stable time sort keeps input order
        expected = np.sum(same & earlier & (t >= t[i] - 3_600))
        assert f["card1_n_1h"].iloc[i] == expected


def test_builder_maps_unseen_categories_to_missing():
    df = build_features(make_synthetic(2000, seed=3))
    s = time_split(df)
    fb = FeatureBuilder().fit(s.train)
    test = s.test.copy()
    test.loc[test.index[0], "card4"] = "brand-new-network"
    x = fb.transform(test)
    assert "card4" in fb.categorical_columns
    assert pd.isna(x["card4"].iloc[0])
    assert list(x.columns) == fb.columns
    assert TARGET not in x.columns and "uid" not in x.columns and "day" not in x.columns


def test_builder_requires_fit():
    with pytest.raises(RuntimeError):
        FeatureBuilder().transform(make_synthetic(10))
