import numpy as np
import pytest
import xgboost as xgb

from ml.src.data import TARGET, make_synthetic, time_split
from ml.src.features import FeatureBuilder, build_features
from ml.src.model import FraudScorer


@pytest.fixture(scope="module")
def scorer_and_rows():
    s = time_split(build_features(make_synthetic(4000, seed=4)))
    builder = FeatureBuilder().fit(s.train)
    model = xgb.XGBClassifier(
        n_estimators=40, max_depth=3, enable_categorical=True, tree_method="hist"
    )
    model.fit(builder.transform(s.train), s.train[TARGET])
    return FraudScorer(model.get_booster(), builder, threshold=0.3), s.test.head(50)


def test_predict_shape_and_decision(scorer_and_rows):
    scorer, rows = scorer_and_rows
    out = scorer.predict(rows)
    assert len(out) == len(rows)
    assert ((out["score"] >= 0.3) == out["is_fraud"]).all()


def test_reasons_are_top_positive_contributions(scorer_and_rows):
    scorer, rows = scorer_and_rows
    reasons = scorer.explain(rows, top_k=3)
    for r in reasons:
        assert len(r) <= 3
        contribs = [x["contribution"] for x in r]
        assert all(c > 0 for c in contribs)
        assert contribs == sorted(contribs, reverse=True)
        assert all(x["feature"] in scorer.builder.columns for x in r)


def test_save_load_roundtrip(scorer_and_rows, tmp_path):
    scorer, rows = scorer_and_rows
    loaded = FraudScorer.load(scorer.save(tmp_path / "s"))
    assert loaded.threshold == scorer.threshold
    np.testing.assert_allclose(loaded.score(rows), scorer.score(rows), rtol=1e-6)
