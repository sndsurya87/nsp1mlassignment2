# Online Shopper Purchase Prediction

M.Tech (AIML), Machine Learning, Assignment 2

## Problem statement

An online retailer sees far more browsing sessions than sales. The goal is to predict, from a
session's browsing behaviour and context, whether it ends in a purchase. This is a binary
classification problem with an imbalanced target: only about 16% of sessions convert.

Five classifiers are implemented on the same dataset and compared on six metrics. The trained
models are served through an interactive Streamlit app that evaluates them on an uploaded test CSV.

## Dataset description

**Source:** [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset),
UCI Machine Learning Repository (Sakar, Polat, Katircioglu and Kastro, *Neural Computing and
Applications*, 2019).

| Property | Value |
|---|---|
| Instances | 12,330 raw; 12,205 after removing 125 exact duplicates |
| Features | 17 (10 numeric, 7 categorical) |
| Target | `Revenue`: whether the session ended in a purchase |
| Class balance | 15.6% positive, after removing duplicates |
| Missing values | None |
| Train / test split | 9,764 / 2,441 rows (80/20, stratified, fixed seed) |

| Feature(s) | Type | Meaning |
|---|---|---|
| `Administrative`, `Informational`, `ProductRelated` | Numeric | Pages of each kind visited in the session |
| `Administrative_Duration`, `Informational_Duration`, `ProductRelated_Duration` | Numeric | Time spent on pages of each kind (seconds) |
| `BounceRates`, `ExitRates` | Numeric | Google Analytics bounce and exit rates of the pages visited |
| `PageValues` | Numeric | Average value of the pages visited before a transaction |
| `SpecialDay` | Numeric | Closeness of the visit to a special day, such as Valentine's Day |
| `Month`, `Weekend` | Categorical | When the session took place |
| `OperatingSystems`, `Browser`, `Region`, `TrafficType` | Categorical | Integer-coded labels for the visitor's context |
| `VisitorType` | Categorical | New, returning or other |

## GitHub repository link

[Git Repo](https://github.com/sndsurya87/nsp1mlassignment2.git)
[Streamlit App](https://nsp1mlassignment2-eqz2pmbyd6vqwvvq9vqcoc.streamlit.app)

## Models used

**Preprocessing**, fitted inside each model's pipeline so that it never sees validation or test data:

- Numeric features are median-imputed. They are also Yeo-Johnson transformed and standardised for
  every model except the Decision Tree, because they are heavily right-skewed
  (`ProductRelated_Duration` has a median of 599 and a maximum of 63,974).
- Categorical features are one-hot encoded, including the integer-coded labels, whose numeric
  order has no meaning. Categories seen fewer than 10 times are pooled.

**Tuning:** each model was grid-searched with stratified 5-fold cross-validation on the training
set, selecting on MCC. The test set was not used for tuning. All models use the default decision
threshold.

| Model | Selected hyper-parameters |
|---|---|
| Logistic Regression | `C=0.003`, `class_weight="balanced"` |
| Decision Tree | `criterion="gini"`, `max_depth=6`, `min_samples_leaf=50` |
| KNN | `n_neighbors=41`, `weights="distance"`, `p=2` |
| Naive Bayes (Gaussian) | `var_smoothing=0.1` |
| Support Vector Machine (RBF) | `C=1.0`, `gamma="scale"` |

### Comparison table

Results on the test set (2,441 rows). Precision, recall and F1 refer to the purchase class. SVM
AUC is computed from its decision function.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8751 | 0.9201 | 0.5717 | 0.8037 | 0.6681 | 0.6069 |
| Decision Tree | 0.9000 | 0.9259 | 0.7197 | 0.5916 | 0.6494 | 0.5957 |
| KNN | 0.8906 | 0.9168 | 0.6910 | 0.5445 | 0.6091 | 0.5518 |
| Naive Bayes | 0.8247 | 0.8737 | 0.4645 | 0.7880 | 0.5845 | 0.5097 |
| Support Vector Machine | 0.9021 | 0.8893 | 0.7173 | 0.6178 | 0.6639 | 0.6092 |

For reference, always predicting "no purchase" scores 0.8435 accuracy, 0.5 AUC and 0 MCC. Accuracy
alone is therefore a weak guide on this dataset, so MCC is the main basis for comparison.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best recall (0.8037) and F1 (0.6681): it finds 307 of the 382 buyers. Balanced class weights cause this, and the cost is the second-lowest precision (230 false alarms). Its MCC is statistically level with the SVM's, which is strong value for a simple linear model. |
| Decision Tree | Best AUC (0.9259) and precision (0.7197). The search preferred a small tree (depth 6, at least 50 samples per leaf), which suggests deeper trees overfit. `PageValues` accounts for 77% of its importance, so it is effectively a few rules around one feature. It is also the easiest model to explain. |
| KNN | Second-lowest MCC (0.5518) and lowest recall (0.5445). It needed a large neighbourhood (k = 41), and the 69 mostly one-hot dimensions likely dilute distance as a measure of similarity. It also had the largest drop from CV MCC to test MCC (0.5784 to 0.5518). It ranks well (AUC 0.9168), but its default-threshold decisions are weaker. |
| Naive Bayes | Lowest on accuracy, AUC, precision, F1 and MCC. Its accuracy (0.8247) is below the majority-class baseline, although its MCC shows it is far from useless. Its assumptions are visibly violated: `BounceRates` and `ExitRates` correlate at 0.90, and one-hot columns are not Gaussian. The large selected smoothing (0.1) is consistent with unstable variance estimates. Its one strength is high recall (0.7880). |
| Support Vector Machine | Best accuracy (0.9021) and MCC (0.6092), with a good balance of precision and recall. Its AUC (0.8893) is only fourth of five, so it makes good decisions at its default threshold but ranks sessions less well than every model except Naive Bayes. It was the slowest to tune despite a small grid (13.2 s for 12 candidates) and keeps 2,610 support vectors. |
| **Overall, winner for your dataset?** | **Support Vector Machine**, on MCC in both cross-validation (0.6063) and testing (0.6092), but only narrowly. Its lead is within sampling noise: the 95% bootstrap interval for its MCC gap includes zero against Logistic Regression (-0.034 to 0.039) and the Decision Tree (-0.016 to 0.043), while it is clearly ahead of KNN and Naive Bayes. The practical choice depends on cost: Logistic Regression if missing a buyer is the costlier error, the Decision Tree if explainability matters most. |

### Limitation: dependence on `PageValues`

`PageValues` carries most of the signal. With that feature neutralised and hyper-parameters left
unchanged, AUC falls from 0.87–0.93 to 0.71–0.76 across the models, and the SVM stops predicting
purchases at all. Because Google Analytics derives page value from completed transactions, it may
partly encode the outcome or be unavailable early in a live session. These results may therefore
overstate what could be achieved in real-time use. This check does not re-tune the models, so it
is a sensitivity test and not an estimate of the best model without the feature.

## Repository structure

```
├── app.py                  Streamlit app
├── requirements.txt        Pinned runtime dependencies
├── test_data.csv           Hold-out test set (raw features and target)
├── data/                   Raw UCI dataset
├── model/                  Five saved pipelines, metrics.json, robustness.json
├── shopper_intent/         Source code
│   ├── config.py           Paths, column schema, constants
│   ├── data.py             Download, validation, train/test split
│   ├── preprocessing.py    Column transformer
│   ├── models.py           Classifiers and search spaces
│   ├── evaluation.py       Metrics and reports
│   ├── train.py            Tuning, evaluation and saving
│   ├── robustness.py       Bootstrap and PageValues checks
│   └── inference.py        Scoring saved models, with no refitting
└── tests/                  Automated tests
```

## Running the project

Requires Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

python -m shopper_intent.train        # tune, evaluate and save all five models
python -m shopper_intent.robustness   # bootstrap and PageValues checks
pytest                                # run the tests
streamlit run app.py                  # start the app
```

## Using the app

The app opens showing results for the bundled `test_data.csv`. Upload your own CSV from the sidebar
to replace it. The file needs all 17 feature columns and `Revenue`. Pick a model from the dropdown
to see its metrics, confusion matrix and classification report. Uploaded data is only ever used for
prediction; the saved models are not retrained.
