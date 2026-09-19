# data/

The dataset is **not** stored in this repository. It is large (about 245 MB) and
belongs to its authors. Download it yourself using the steps below.

## Dataset: WELFake

| | |
|---|---|
| **Name** | WELFake dataset for fake news detection in text data |
| **Authors** | Pawan Kumar Verma, Prateek Agrawal, Radu Prodan |
| **Source** | Zenodo: https://zenodo.org/records/4561253 |
| **DOI** | [10.5281/zenodo.4561253](https://doi.org/10.5281/zenodo.4561253) |
| **License** | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| **File** | `WELFake_Dataset.csv` (about 245 MB) |
| **MD5 (from Zenodo)** | `73c9675a4b3d09f86a6933d0b8d7d908` |
| **Paper** | Verma, P. K., Agrawal, P., & Prodan, R. (2021). *WELFake: Word Embedding Over Linguistic Features for Fake News Detection.* IEEE Transactions on Computational Social Systems. https://doi.org/10.1109/TCSS.2021.3068519 (citation as given on the Zenodo page) |

According to its Zenodo page, WELFake merges four earlier collections (Kaggle,
McIntire, Reuters and BuzzFeed Political) into 72,134 articles.

**Attribution (CC BY 4.0 requires it):** NewsLens is trained on the WELFake dataset
by Verma, Agrawal and Prodan, available at https://zenodo.org/records/4561253 under
CC BY 4.0. NewsLens does not redistribute the dataset. It only uses it to train the
model. The MIT license of this repository covers the NewsLens code, not the dataset.

## Download and placement

1. Open https://zenodo.org/records/4561253 in your browser.
2. Click **Download** next to `WELFake_Dataset.csv`.
3. Move the file into this `data/` folder with exactly this name:

   ```
   NewsLens/
   └── data/
       └── WELFake_Dataset.csv
   ```

   If your browser saved it as `WELFake_Dataset (1).csv`, rename it.
4. Optional: check that the file is complete.

   ```bash
   # macOS
   md5 data/WELFake_Dataset.csv
   # Linux
   md5sum data/WELFake_Dataset.csv
   # Windows (PowerShell)
   Get-FileHash data\WELFake_Dataset.csv -Algorithm MD5
   ```
   Expected: `73c9675a4b3d09f86a6933d0b8d7d908` (Windows prints it in upper case).

`data/*.csv` is listed in `.gitignore`, so the file will not be pushed to GitHub by accident.

## Expected CSV columns

| Column | Meaning |
|---|---|
| *(unnamed)* | Serial number starting at 0. Ignored. |
| `title` | Headline. Missing in 558 rows. |
| `text` | Article body. Missing in 39 rows. |
| `label` | `0` or `1`, see below. |

pandas reads **72,134 rows**. The Zenodo page notes that the file has 78,098 lines
but only 72,134 entries load into a data frame, because many articles contain
line breaks inside quoted text. That is expected.

The adapter in `src/data.py` (`load_raw_csv`) lower-cases column names, checks that
`title`, `text` and `label` exist, converts the label to a number, drops rows with an
unknown label, and maps it to the project's internal labels.

## Label mapping (important)

The Zenodo page describes the label as "0 = fake and 1 = real". **The file itself
behaves the other way round**, and NewsLens uses:

| Raw `label` | Meaning used by NewsLens | Internal label |
|---|---|---|
| `0` | real | `0` (real) |
| `1` | fake | `1` (fake, the positive class) |

Evidence, measured on the file with the MD5 above (`python -m src.inspect_dataset`):

* **60.7%** of articles with raw label `0` contain a "(Reuters)" dateline, versus
  **0.0%** with raw label `1`. The Reuters part of WELFake comes from real news
  articles, so raw `0` must be real.
* Raw label `1` has **37,106** rows and raw label `0` has **35,028** rows. The Zenodo
  description gives 37,106 *fake* and 35,028 *real* articles, which again means `1` = fake.

You can re-run the check any time:

```bash
python -m src.inspect_dataset
```

It prints `consistent` when the mapping in `src/config.py` (`RAW_LABEL_MAP`) matches
the Reuters evidence.
