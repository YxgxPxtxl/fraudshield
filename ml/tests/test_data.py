import pytest

from ml.src.data import ID_COL, TARGET, TIME_COL, make_synthetic, numeric_features, time_split


@pytest.fixture
def df():
    # Shuffle so the test proves time_split sorts rather than relying on input order.
    return make_synthetic(2000, seed=1).sample(frac=1, random_state=0)


def test_split_sizes_sum_to_total(df):
    s = time_split(df)
    assert len(s.train) + len(s.val) + len(s.test) == len(df)
    assert len(s.train) == pytest.approx(0.7 * len(df), abs=2)


def test_split_is_chronological_no_overlap(df):
    s = time_split(df)
    assert s.train[TIME_COL].max() <= s.val[TIME_COL].min()
    assert s.val[TIME_COL].max() <= s.test[TIME_COL].min()


def test_split_has_no_shared_ids(df):
    s = time_split(df)
    ids = [set(part[ID_COL]) for part in (s.train, s.val, s.test)]
    assert not (ids[0] & ids[1]) and not (ids[1] & ids[2]) and not (ids[0] & ids[2])


@pytest.mark.parametrize("val,test", [(0, 0.15), (0.5, 0.5), (0.15, 1.2)])
def test_split_rejects_bad_fractions(df, val, test):
    with pytest.raises(ValueError):
        time_split(df, val, test)


def test_numeric_features_exclude_leaky_columns(df):
    feats = numeric_features(df)
    assert TARGET not in feats and ID_COL not in feats and TIME_COL not in feats
    assert "TransactionAmt" in feats
    assert "ProductCD" not in feats  # categorical, handled in week 2


def test_synthetic_fraud_rate_is_realistic():
    rate = make_synthetic(20_000, fraud_rate=0.035)[TARGET].mean()
    assert 0.025 < rate < 0.045
