"""NewsLens - Streamlit web app.

Run locally with:   streamlit run app.py

The app only LOADS a model that was trained beforehand with `python -m src.train`.
It never trains on startup, never falls back to a dummy model, and never
stores or logs the text that users paste.
"""

from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from src import config
from src.charts import confusion_matrix_fig, contributions_bar, probability_bar
from src.examples import EXAMPLES
from src.inference import ModelNotFoundError, explain_terms, load_bundle, predict
from src.validation import validate_input

st.set_page_config(page_title="NewsLens - Fake News Detection", page_icon="📰",
                   layout="centered")

TEXT_KEY = "news_text"
RESULT_KEY = "last_result"

DISCLAIMER = (
    "**Educational classifier, not a fact-checking service.** NewsLens recognises "
    "writing patterns it learned from a labeled training dataset. It does **not** check "
    "facts, search the web, or read current news, and it cannot prove that an article "
    "is true or false."
)


# ---------------------------------------------------------------------------
# Model loading (cached: loaded once per server process, not per interaction)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the trained model...")
def get_bundle(model_dir: str):
    return load_bundle(config.get_model_dir() if not model_dir else model_dir)


def _text_hash(text: str) -> str:
    # Only a hash is kept, to know whether the shown result matches the text box.
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def load_example(name: str) -> None:
    st.session_state[TEXT_KEY] = EXAMPLES[name]
    st.session_state.pop(RESULT_KEY, None)


def reset() -> None:
    st.session_state[TEXT_KEY] = ""
    st.session_state.pop(RESULT_KEY, None)


# ---------------------------------------------------------------------------
# UI sections
# ---------------------------------------------------------------------------
def header() -> None:
    st.markdown(
        "<h1 style='margin-bottom:0'>📰 NewsLens</h1>"
        "<p style='font-size:1.1rem;opacity:.8;margin-top:.2rem'>"
        "Fake news detection using machine learning</p>",
        unsafe_allow_html=True,
    )
    st.write(
        "Paste an English news headline or article. A TF-IDF + Logistic Regression model "
        "trained on the public WELFake dataset estimates whether its writing looks more "
        "like the *real* or the *fake* articles it was trained on."
    )
    st.info(DISCLAIMER, icon="ℹ️")


def render_result(result: dict, bundle) -> None:
    pred = result["prediction"]
    is_fake = pred.label == config.LABEL_FAKE
    color = "#E0762B" if is_fake else "#2E6FD8"
    icon = "⚠️" if is_fake else "✅"

    with st.container(border=True):
        st.markdown(
            f"<div style='border-left:6px solid {color};padding:.25rem 0 .25rem 1rem'>"
            f"<div style='font-size:.9rem;opacity:.75'>Predicted label</div>"
            f"<div style='font-size:1.8rem;font-weight:700;color:{color}'>{icon} "
            f"{pred.display_label}</div>"
            f"<div style='font-size:.95rem;opacity:.85'>Model probability estimate: "
            f"<b>{pred.prob_fake:.0%}</b> fake / <b>{pred.prob_real:.0%}</b> real</div></div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(probability_bar(pred.prob_real, pred.prob_fake),
                        width="stretch", config={"displayModeBar": False})
        st.caption(
            "The percentage is the model's **estimate** of how closely the text matches the "
            "fake vs real articles in its training data. It is **not** the probability that "
            "the news is factually true or false."
        )

        strength = pred.confidence
        if strength >= 0.9:
            level = "strongly"
        elif strength >= 0.7:
            level = "moderately"
        else:
            level = "only weakly"
        st.markdown(
            f"**How to read this:** the writing style and vocabulary of this text "
            f"{level} resemble the *{pred.label_name}* articles in the WELFake training data. "
            "Treat it as a prompt to check the story yourself: look for the original source, "
            "the author, the date and whether trusted outlets report the same facts."
        )
        if 0.4 <= pred.prob_fake <= 0.6:
            st.warning("The estimate is close to 50%, so the model is very uncertain here.")

    for w in result["warnings"]:
        st.warning(w, icon="⚠️")

    terms = result.get("terms")
    if terms and (terms["toward_fake"] or terms["toward_real"]):
        with st.expander("Which terms influenced this prediction?"):
            st.write(
                "For Logistic Regression each term's contribution is exactly its TF-IDF "
                "weight in this text × the model's learned coefficient. Orange bars pushed "
                "toward *fake*, blue bars toward *real*."
            )
            st.plotly_chart(contributions_bar(terms), width="stretch",
                            config={"displayModeBar": False})
            st.caption(
                "These are **statistical signals the model learned, not evidence**. A term "
                "pushing toward 'fake' does not make a statement false; it only means the "
                "term was more common in the fake-labeled training articles."
            )


def analyze_tab(bundle) -> None:
    st.markdown("##### Try an illustrative example")
    st.caption("Written for this demo. These are not real news stories and have no "
               "true/false label.")
    cols = st.columns(len(EXAMPLES))
    for col, name in zip(cols, EXAMPLES):
        col.button(name, on_click=load_example, args=(name,), width="stretch")

    text = st.text_area(
        "News headline or article (English)", key=TEXT_KEY, height=260,
        placeholder="Paste a headline, or better, a full article here...",
        help=f"Between {config.MIN_WORDS} words and {config.MAX_CHARS:,} characters. "
             "Full articles give more reliable results than headlines.",
    )
    words = len((text or "").split())
    st.caption(f"{words} words · {len(text or ''):,} / {config.MAX_CHARS:,} characters")

    c1, c2, _ = st.columns([1.3, 1, 2])
    analyze = c1.button("🔍 Analyze News", type="primary", width="stretch")
    c2.button("↺ Reset", on_click=reset, width="stretch")

    if analyze:
        check = validate_input(text)
        if not check.ok:
            st.session_state.pop(RESULT_KEY, None)
            st.error(check.error, icon="🚫")
        else:
            with st.spinner("Analyzing the text..."):
                try:
                    prediction = predict(bundle.pipeline, text)
                    terms = explain_terms(bundle.pipeline, text)
                except Exception as exc:  # show a friendly message, keep the app alive
                    st.error(f"Something went wrong while analysing the text: {exc}")
                    return
            st.session_state[RESULT_KEY] = {
                "prediction": prediction, "terms": terms,
                "warnings": check.warnings, "text_hash": _text_hash(text),
            }

    result = st.session_state.get(RESULT_KEY)
    if result:
        if result["text_hash"] != _text_hash(text or ""):
            st.caption("ℹ️ The text has changed since this result. Click **Analyze News** "
                       "again to update it.")
        render_result(result, bundle)


def performance_tab(bundle) -> None:
    metrics = bundle.metrics
    if not metrics or metrics.get("status") != "evaluated":
        st.info("**Not yet evaluated.** No saved evaluation results were found. Run "
                "`python -m src.train` to train the model and create `models/metrics.json`.")
        return

    test = metrics["test"]
    st.markdown(f"#### Held-out test set · {metrics['final_model']}")
    st.caption(f"Evaluated once on {test['n_samples']:,} test articles that were never used "
               f"for training or model selection. Trained {metrics['trained_at_utc'][:10]}.")
    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{test['accuracy']:.1%}")
    cols[1].metric("Precision (fake)", f"{test['precision_fake']:.1%}")
    cols[2].metric("Recall (fake)", f"{test['recall_fake']:.1%}")
    cols[3].metric("F1 (fake)", f"{test['f1_fake']:.1%}")

    st.markdown("##### Confusion matrix (test set)")
    cm = test["confusion_matrix"]
    st.plotly_chart(confusion_matrix_fig(cm["matrix"], cm["labels"]), width="stretch",
                    config={"displayModeBar": False})

    head = metrics.get("test_headlines_only")
    if head:
        st.markdown("##### Headlines only (same test articles, body text removed)")
        st.write(
            f"Accuracy **{head['accuracy']:.1%}**, F1 (fake) **{head['f1_fake']:.1%}** on "
            f"{head['n_samples']:,} headlines, compared with **{test['accuracy']:.1%}** "
            "accuracy on full articles. This is why short inputs get a warning."
        )

    st.markdown("##### Model selection on the validation set")
    rows = [{
        "Model": c["name"],
        "Role": "Main model" if c["kind"] == "logistic_regression" else "Baseline",
        "Val accuracy": c["validation"]["accuracy"],
        "Val macro F1": c["validation"]["f1_macro"],
        "Val F1 (fake)": c["validation"]["f1_fake"],
    } for c in metrics["validation_candidates"]]
    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.format({"Val accuracy": "{:.2%}", "Val macro F1": "{:.2%}",
                         "Val F1 (fake)": "{:.2%}"}),
        hide_index=True, width="stretch", height=38 * (len(df) + 1) + 3,
    )
    st.caption(f"The Logistic Regression setting with the best validation macro F1 was "
               f"selected, refit on train + validation and tested once. Best baseline: "
               f"{metrics['baseline_model']}.")

    s = metrics["dataset_stats"]
    with st.expander("Dataset preparation details"):
        st.write({
            "Rows in CSV (as parsed)": s["rows_raw"],
            "Dropped: empty or under 5 words": s["dropped_empty_or_tiny"],
            "Dropped: exact duplicates": s["dropped_exact_duplicates"],
            "Dropped: duplicates with conflicting labels": s["dropped_conflicting_label_duplicates"],
            "Rows after cleaning": s["rows_after_cleaning"],
            "Near-duplicate groups": s["near_duplicate_groups"],
            "Train / validation / test rows": f"{s['train_rows']} / {s['validation_rows']} / "
                                               f"{s['test_rows']}",
        })

    st.warning(
        "These numbers describe performance on articles from the **same dataset** the model "
        "was trained on. Real-world accuracy on new, recent or differently sourced news is "
        "expected to be lower and has not been measured.", icon="⚠️")


def about_tab(bundle) -> None:
    meta = bundle.metadata
    ds = meta.get("dataset", {})
    st.markdown(f"""
#### How it works
1. **Cleaning.** URLs, social-media handles and publisher markers such as the
   "(Reuters)" dateline are removed so the model can't simply learn which outlet
   wrote an article.
2. **TF-IDF.** The text becomes a vector of word and two-word-phrase weights. Words that
   are frequent in *this* text but rare across the training articles get high weights.
3. **Logistic Regression.** Each term has a learned weight. The weighted sum is turned
   into a 0–100% estimate. Above 50% → *Likely fake*, otherwise *Likely real*.
4. **Baseline.** Multinomial Naive Bayes was trained and compared on the validation set.

#### Model
- **Model:** {meta.get('model_name', 'unknown')} · vocabulary {meta.get('vocabulary_size', '?'):,} terms
- **Trained:** {meta.get('trained_at_utc', 'unknown')} (scikit-learn {meta.get('sklearn_version', '?')})

#### Dataset
- **{ds.get('name', 'unknown')}** by {ds.get('authors', 'unknown')}
- Source: {ds.get('source_url', '')} · DOI {ds.get('doi', '')}
- License: {ds.get('license', 'unknown')}

#### Limitations
- It learns **style and vocabulary**, not truth. A false claim written in a sober style
  can be labeled *real*, and a true but emotional article can be labeled *fake*.
- The training data is dominated by **US political news from around the 2016 US
  election** (terms like "hillary", "obama" and "2016" are among the strongest signals).
  Newer topics, names and events are unfamiliar to the model.
- Some learned signals are about **publisher style**, not content, for example
  wire-service phrasing such as "said on Tuesday" (→ real) or "via", "video" and
  "watch" (→ fake). Obvious publisher markers are removed, but not all of them.
- Labels in WELFake mostly reflect the **source** an article came from, not a
  fact-check of each individual claim.
- **Short headlines** are less reliable than full articles.
- English only.

#### Privacy
Text you paste is processed in memory to make the prediction. NewsLens does not save it
to disk or write it to logs.
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    header()
    try:
        bundle = get_bundle(str(config.get_model_dir()))
    except ModelNotFoundError as exc:
        st.error("**The trained model is missing, so NewsLens can't make predictions.**",
                 icon="🚫")
        st.code(str(exc), language="text")
        st.stop()
    except Exception as exc:
        st.error(f"**The model could not be loaded:** {exc}\n\nMake sure the packages in "
                 "requirements.txt are installed with the pinned versions.", icon="🚫")
        st.stop()

    if bundle.metadata.get("is_test_fixture"):
        st.warning("A tiny **test fixture** model is loaded. Its predictions are meaningless. "
                   "Train the real model with `python -m src.train`.", icon="🧪")

    tab1, tab2, tab3 = st.tabs(["🔍 Analyze", "📊 Model Performance", "ℹ️ About the Model"])
    with tab1:
        analyze_tab(bundle)
    with tab2:
        performance_tab(bundle)
    with tab3:
        about_tab(bundle)

    st.divider()
    st.caption("NewsLens is an educational machine-learning project. Always verify news with "
               "trusted sources and professional fact-checkers.")


main()
