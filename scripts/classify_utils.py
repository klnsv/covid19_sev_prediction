"""Helpers for patient-level classification (CONTEXT.md section 5.6, contract C5).

Reads nothing and writes nothing: steps 05 and 11 import these functions.
    make_model(name, seed, pca, n_plain)                  -> sklearn Pipeline
    run_cv(X, y, model_name, cv_scheme, seed, pca, n_plain) -> out-of-fold scores
    get_score(model, X)                                   -> P(severe) or SVM decision value
    auroc_balacc(y, score)                                -> (AUROC, balanced accuracy)
    permutation_null(...) / perm_pvalue(...)              -> null AUROCs / permutation p

X is a samples x features table with one row per PATIENT. All preprocessing
(gene filter, scaling, PCA) lives inside the pipeline, so it is fit on the training fold only.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

MODEL_NAMES = ["logreg", "linsvm", "rf"]
CV_SCHEMES = ["rep5fold", "lopo"]


class TopVarianceGenes(BaseEstimator, TransformerMixin):
    """Keep the n_genes columns with the highest variance in the TRAINING fold.

    This is the in-fold HVG selection for the pseudobulk features: the gene
    choice is made inside fit(), so test patients never influence it.
    """

    def __init__(self, n_genes=2000):
        self.n_genes = n_genes

    def fit(self, X, y=None):
        self.keep_ = np.argsort(X.var(axis=0))[::-1][:self.n_genes]
        return self

    def transform(self, X):
        return X[:, self.keep_]


def make_model(name, seed, pca=False, n_plain=0):
    """Fixed hyperparameters (no tuning in the sprint).

    pca=True is for the high-dimensional pseudobulk features: top-variance genes
    -> scale -> PCA (10 components), all fit on the training fold only.
    n_plain > 0 (combined features): the first n_plain columns (composition) are
    only scaled; the remaining columns (pseudobulk) go through the PCA branch.
    """
    if name == "logreg":
        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=5000, random_state=seed)
    elif name == "linsvm":
        clf = LinearSVC(C=1.0, class_weight="balanced", max_iter=20000,
                        random_state=seed)
    elif name == "rf":
        clf = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                     random_state=seed, n_jobs=1)
    else:
        raise ValueError("unknown model: " + name)
    if pca and n_plain > 0:
        n_cols_plain = list(range(n_plain))
        pb_branch = make_pipeline(TopVarianceGenes(2000), StandardScaler(),
                                  PCA(n_components=10, random_state=seed))
        # pseudobulk columns = everything after the first n_plain columns
        both = ColumnTransformer([("comp", StandardScaler(), n_cols_plain),
                                  ("pb", pb_branch, lambda X: list(range(n_plain, X.shape[1])))])
        return make_pipeline(both, clf)
    if pca:
        return make_pipeline(TopVarianceGenes(2000), StandardScaler(),
                             PCA(n_components=10, random_state=seed), clf)
    return make_pipeline(StandardScaler(), clf)


def get_score(model, X):
    """P(severe) if the model has it, else the decision value (SVM)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def run_cv(X, y, model_name, cv_scheme, seed, pca=False, n_plain=0):
    """Return one out-of-fold score per patient (same order as y).

    rep5fold: one stratified 5-fold split with this seed (call with seeds 0-9
              for the 10 repeats).
    lopo:     leave-one-patient-out (seed only sets the model's randomness).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    if cv_scheme == "rep5fold":
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        splits = splitter.split(X, y)
    elif cv_scheme == "lopo":
        splits = LeaveOneOut().split(X)
    else:
        raise ValueError("unknown cv_scheme: " + cv_scheme)

    oof = np.zeros(len(y))
    for train_idx, test_idx in splits:
        model = make_model(model_name, seed, pca, n_plain)
        model.fit(X[train_idx], y[train_idx])
        oof[test_idx] = get_score(model, X[test_idx])
    return oof


def auroc_balacc(y, score):
    """AUROC on pooled OOF scores; balanced accuracy at the natural threshold.

    The threshold is 0.5 for probabilities and 0 for SVM decision values.
    We detect which one by the score range.
    """
    y = np.asarray(y, dtype=int)
    score = np.asarray(score)
    auroc = roc_auc_score(y, score)
    threshold = 0.5 if (score.min() >= 0 and score.max() <= 1) else 0.0
    bal_acc = balanced_accuracy_score(y, (score > threshold).astype(int))
    return auroc, bal_acc


def permutation_null(X, y, model_name, cv_scheme, n_perm, seed, pca=False, n_plain=0):
    """Shuffle labels across patients, rerun the whole CV, collect AUROCs."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y, dtype=int)
    null_aurocs = []
    for i in range(n_perm):
        y_shuffled = rng.permutation(y)
        oof = run_cv(X, y_shuffled, model_name, cv_scheme, i % 10, pca, n_plain)
        null_aurocs.append(roc_auc_score(y_shuffled, oof))
    return null_aurocs


def perm_pvalue(observed, null_aurocs):
    """p = (1 + #null >= observed) / (1 + n_perm)."""
    n_ge = sum(a >= observed for a in null_aurocs)
    return (1 + n_ge) / (1 + len(null_aurocs))
