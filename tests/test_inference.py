import joblib
import pandas as pd
import pytest

from src import config
from src.data import DatasetError, load_raw_csv
from src.inference import ModelNotFoundError, explain_terms, load_bundle, predict


def test_prediction_structure_and_label_mapping(toy_pipeline):
    pred = predict(toy_pipeline, "SHOCKING secret they don't want you to know, share now!!!")
    assert pred.label in (config.LABEL_REAL, config.LABEL_FAKE)
    assert pred.label_name == config.LABEL_NAMES[pred.label]
    assert pred.display_label in ("Likely real", "Likely fake")
    assert 0.0 <= pred.prob_fake <= 1.0
    assert pred.prob_fake + pred.prob_real == pytest.approx(1.0)
    # Label must agree with the probability and with the pipeline's own predict().
    assert (pred.label == config.LABEL_FAKE) == (pred.prob_fake >= 0.5)
    assert pred.label == toy_pipeline.predict(
        ["SHOCKING secret they don't want you to know, share now!!!"])[0]


def test_toy_model_separates_its_own_training_styles(toy_pipeline):
    # Plumbing check only: the toy model should at least fit its own toy data.
    assert predict(toy_pipeline, "shocking secret share it now before deleted!!!").label_name == "fake"
    assert predict(toy_pipeline, "The committee approved the budget, officials said.").label_name == "real"


def test_explain_terms_matches_decision_function(toy_pipeline):
    text = "shocking secret the committee approved"
    terms = explain_terms(toy_pipeline, text, top_n=1000)
    total = terms["intercept"] + sum(v for _, v in terms["toward_fake"]) + \
        sum(v for _, v in terms["toward_real"])
    assert total == pytest.approx(terms["total_log_odds"], abs=1e-5)
    assert all(v > 0 for _, v in terms["toward_fake"])
    assert all(v < 0 for _, v in terms["toward_real"])


def test_save_and_reload_gives_identical_predictions(tmp_path, toy_pipeline):
    path = tmp_path / "model.joblib"
    joblib.dump(toy_pipeline, path, compress=3)
    reloaded = joblib.load(path)
    texts = ["the committee approved the budget", "shocking secret share now!!!"]
    assert (reloaded.predict_proba(texts) == toy_pipeline.predict_proba(texts)).all()


def test_load_bundle_from_folder(fixture_model_dir):
    bundle = load_bundle(fixture_model_dir)
    assert bundle.metadata["is_test_fixture"] is True
    assert bundle.metrics is None
    assert predict(bundle.pipeline, "the committee approved the budget today").label_name in ("real", "fake")


def test_missing_model_raises_clear_error(tmp_path):
    with pytest.raises(ModelNotFoundError) as exc:
        load_bundle(tmp_path)
    msg = str(exc.value)
    assert "python -m src.train" in msg
    assert "data/README.md" in msg


def test_dataset_adapter_maps_labels_and_reports_missing_file(tmp_path):
    with pytest.raises(DatasetError):
        load_raw_csv(tmp_path / "nope.csv")

    csv = tmp_path / "mini.csv"
    pd.DataFrame({
        "Unnamed: 0": [0, 1, 2],
        "title": ["a", "b", "c"],
        "text": ["x", "y", "z"],
        "label": [0, 1, "oops"],
    }).to_csv(csv, index=False)
    df = load_raw_csv(csv)
    assert list(df.columns) == ["title", "text", "label"]
    assert df["label"].tolist() == [config.RAW_LABEL_MAP[0], config.RAW_LABEL_MAP[1]]
    assert df.attrs["dropped_bad_label"] == 1


def test_dataset_adapter_rejects_wrong_schema(tmp_path):
    csv = tmp_path / "wrong.csv"
    pd.DataFrame({"headline": ["a"], "body": ["b"]}).to_csv(csv, index=False)
    with pytest.raises(DatasetError, match="missing required column"):
        load_raw_csv(csv)
