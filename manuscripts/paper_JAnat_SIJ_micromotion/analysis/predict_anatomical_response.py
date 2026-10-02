"""Internal subject-held-out prediction of archived SIJ kinematics.

Fixed retrospective predictor sets, identical sex-stratified subject folds for
every load/outcome/model, training-only standardization and residual smearing.
Paired bootstrap intervals condition on the saved CV predictions. No FE solves.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

PAPER = Path(__file__).resolve().parents[1]
OUT = PAPER / "tables/prediction"
LOADS = ["SP2leg", "SP1leg", "LAB_phase1", "LAB_phase2", "LAB_phase3"]
OUTCOMES = ["rot_mag_deg", "trans_mag_mm"]
TRIAD = ["AP", "BiischiadicWidth", "SubpubicAngle"]
MORPH = ["AP", "BiacetabularWidth", "BiischiadicWidth", "BituberousWidth",
         "IliopectinealEminenceWidth", "PIT", "SacralWidth", "SubpubicAngle"]
MODELS = {"size": ["age", "log_total_volume"],
          "triad": ["age", "log_total_volume", *TRIAD],
          "triad_sex": ["age", "log_total_volume", *TRIAD, "sex_F", "sex_log_volume"],
          "all_dimensions": ["age", "log_total_volume", *MORPH]}
SEED = 20261002
REPEATS = 10
FOLDS = 10


def predict_log_ols(x_train, y_train, x_test):
    """Return arithmetic-scale predictions with training-only Duan smearing."""
    mean = x_train.mean(axis=0)
    sd = x_train.std(axis=0, ddof=1)
    sd[sd == 0] = 1
    a = np.column_stack([np.ones(len(x_train)), (x_train-mean)/sd])
    b = np.column_stack([np.ones(len(x_test)), (x_test-mean)/sd])
    coef = np.linalg.lstsq(a, np.log(y_train), rcond=None)[0]
    residual = np.log(y_train)-a@coef
    smear = np.mean(np.exp(residual))
    return np.exp(b@coef)*smear


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(PAPER / "tables/generated/subject_base.csv").sort_values("subject_idx")
    data = pd.read_csv(PAPER / "tables/generated/subject_level.csv")
    assert base.subject_idx.tolist() == list(range(len(base)))
    assert len(base) == 278 and len(data) == len(base)*len(LOADS)
    assert not data.duplicated(["subject_idx", "load_case"]).any()
    base["sex_log_volume"] = base.sex_F*base.log_total_volume
    target = np.empty((len(base), len(LOADS), len(OUTCOMES)))
    for j, load in enumerate(LOADS):
        sub = data.loc[data.load_case == load].set_index("subject_idx").loc[base.subject_idx]
        target[:, j] = sub[OUTCOMES]
    assert np.isfinite(target).all() and (target > 0).all()
    predictions = np.full((len(MODELS), REPEATS, *target.shape), np.nan)
    baseline = np.full((REPEATS, *target.shape), np.nan)
    fold_rows = []
    for repeat in range(REPEATS):
        split = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED+repeat)
        for fold, (train, test) in enumerate(split.split(base, base.sex)):
            assert not set(train).intersection(test)
            fold_rows.extend(dict(subject_idx=int(i), repeat=repeat, fold=fold) for i in test)
            for j, load in enumerate(LOADS):
                for k, outcome in enumerate(OUTCOMES):
                    y_train = target[train, j, k]
                    baseline[repeat, test, j, k] = y_train.mean()
                    for m, (_, features) in enumerate(MODELS.items()):
                        x = base[features].to_numpy(float)
                        predictions[m, repeat, test, j, k] = predict_log_ols(x[train], y_train, x[test])
    assert np.isfinite(predictions).all()
    pd.DataFrame(fold_rows).to_csv(OUT / "subject_folds.csv", index=False)
    np.savez_compressed(OUT / "held_out_predictions.npz", predictions=predictions,
                        target=target, baseline=baseline, model_names=np.array(list(MODELS)),
                        loads=np.array(LOADS), outcomes=np.array(OUTCOMES))
    rows = []
    gains = []
    rng = np.random.default_rng(SEED)
    bootstrap = rng.integers(0, len(base), size=(3000, len(base)))
    for j, load in enumerate(LOADS):
        for k, outcome in enumerate(OUTCOMES):
            y = target[:, j, k]
            errors = predictions[:, :, :, j, k]-y[None, None, :]
            mse_subject = np.mean(errors**2, axis=1)
            for m, name in enumerate(MODELS):
                mse = mse_subject[m].mean()
                rows.append(dict(load_case=load, outcome=outcome, model=name,
                                 cv_r2=1-mse/np.var(y, ddof=0), rmse=float(np.sqrt(mse)),
                                 mae=float(np.mean(abs(errors[m]))),
                                 p95_absolute_error=float(np.quantile(abs(errors[m]), .95)),
                                 baseline_rmse=float(np.sqrt(np.mean((baseline[:, :, j, k]-y)**2))),
                                 n_subjects=len(base), repeats=REPEATS, folds=FOLDS))
            for before, after in [("size", "triad"), ("triad", "triad_sex"), ("triad", "all_dimensions")]:
                a, b = list(MODELS).index(before), list(MODELS).index(after)
                improvement = np.sqrt(mse_subject[a][bootstrap].mean(axis=1))-np.sqrt(mse_subject[b][bootstrap].mean(axis=1))
                gains.append(dict(load_case=load, outcome=outcome, comparison=f"{before}_to_{after}",
                                  delta_rmse=float(np.sqrt(mse_subject[a].mean())-np.sqrt(mse_subject[b].mean())),
                                  ci95_low=float(np.quantile(improvement, .025)),
                                  ci95_high=float(np.quantile(improvement, .975)),
                                  reduction_pct=float(100*(1-np.sqrt(mse_subject[b].mean()/mse_subject[a].mean())))))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "prediction_metrics.csv", index=False)
    pd.DataFrame(gains).to_csv(OUT / "paired_prediction_gains.csv", index=False)
    config = dict(seed=SEED, repeats=REPEATS, folds=FOLDS, n_subjects=len(base), models=MODELS,
                  fit="load-specific OLS on log outcome; training-only standardization and smearing",
                  validation="same subject folds for all load cases, outcomes and predictor sets",
                  bootstrap="3000 paired subject resamples of repeat-averaged squared held-out errors; no refitting",
                  selection="retrospective fixed triad; its original selection is not nested within CV",
                  generalization="new subjects in represented cohort/loading configurations; no external validation")
    (OUT / "prediction_protocol.json").write_text(json.dumps(config, indent=2))
    print(metrics.loc[metrics.model.isin(["size", "triad"]), ["load_case", "outcome", "model", "cv_r2", "rmse"]].to_string(index=False), flush=True)
    print(pd.DataFrame(gains).to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
