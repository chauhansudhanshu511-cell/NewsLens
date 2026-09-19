# models/

`python -m src.train` writes three files here:

| File | What it is |
|------|------------|
| `newslens_pipeline.joblib` | The **complete fitted scikit-learn Pipeline**: text cleaning → TF-IDF → Logistic Regression. Training and the app use this same object, so preprocessing is always identical. |
| `metrics.json` | Evaluation results: every validation candidate (Logistic Regression and the Naive Bayes baseline), the final held-out **test** metrics, a headlines-only diagnostic, dataset preparation statistics and the most influential terms. |
| `metadata.json` | Dataset identity (name, DOI, license, MD5 of your CSV), raw → internal label mapping, model name and hyperparameters, training date, Python and scikit-learn versions. |

## Deploying the model

These files are small enough to commit to Git (the pipeline is a few MB, far below
GitHub's 100 MB file limit), so the simplest deployment is:

```bash
git add models/newslens_pipeline.joblib models/metrics.json models/metadata.json
git commit -m "Add trained model"
git push
```

Streamlit Community Cloud then loads them straight from your repository. The app
**never trains on startup**. If the files are missing it shows an error with
instructions instead of making up predictions.

## Security: only load trusted joblib files

A `.joblib` file is a Python pickle. **Loading one can run arbitrary code.** Only
load model files you trained yourself or that come from a source you fully trust
(such as your own repository). Never load a `.joblib` file downloaded from an
unknown website, email or chat.

## Version pinning

A pipeline saved with one scikit-learn version may not load, or may behave
differently, with another. `requirements.txt` pins the exact versions used for
training, and the app warns if the installed scikit-learn differs from the one in
`metadata.json`. If you upgrade scikit-learn, retrain and commit the new artifacts.
