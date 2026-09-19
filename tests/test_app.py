"""Smoke tests for the Streamlit app using Streamlit's built-in AppTest."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(autouse=True)
def _clear_streamlit_cache():
    import streamlit as st
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def test_app_shows_error_when_model_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("NEWSLENS_MODEL_DIR", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("model is missing" in e.value for e in at.error)
    assert len(at.button) == 0          # no Analyze button -> no fake predictions


def test_app_predicts_with_fixture_model(fixture_model_dir, monkeypatch):
    monkeypatch.setenv("NEWSLENS_MODEL_DIR", str(fixture_model_dir))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("test fixture" in w.value for w in at.warning)

    at.text_area(key="news_text").input(
        "The committee approved the annual budget on Tuesday after a public hearing, "
        "officials said in a statement."
    ).run()
    analyze = next(b for b in at.button if "Analyze" in b.label)
    analyze.click().run()
    assert not at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "Predicted label" in html
    assert ("Likely real" in html) or ("Likely fake" in html)
    # Performance tab reports missing metrics honestly.
    assert any("Not yet evaluated" in i.value for i in at.info)


def test_app_rejects_empty_input(fixture_model_dir, monkeypatch):
    monkeypatch.setenv("NEWSLENS_MODEL_DIR", str(fixture_model_dir))
    at = AppTest.from_file(APP, default_timeout=30).run()
    next(b for b in at.button if "Analyze" in b.label).click().run()
    assert any("paste" in e.value.lower() for e in at.error)


def test_reset_clears_text(fixture_model_dir, monkeypatch):
    monkeypatch.setenv("NEWSLENS_MODEL_DIR", str(fixture_model_dir))
    at = AppTest.from_file(APP, default_timeout=30).run()
    next(b for b in at.button if "Neutral" in b.label).click().run()
    assert at.text_area(key="news_text").value
    next(b for b in at.button if "Reset" in b.label).click().run()
    assert at.text_area(key="news_text").value == ""
