"""Independent numerical checks of the revised prediction and mechanism results.

Refits held-out regressions in raw units with statsmodels, rather than the
training-standardized NumPy implementation. Checks all archived mixture
endpoints, triangle inequality, summary errors and paired cohort structure.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

PAPER = Path(__file__).resolve().parents[1]
OUT = PAPER / 'tables/prediction'
archive = np.load(OUT / 'held_out_predictions.npz')
protocol = json.loads((OUT / 'prediction_protocol.json').read_text())
base = pd.read_csv(PAPER / 'tables/generated/subject_base.csv').sort_values('subject_idx')
base['sex_log_volume'] = base.sex_F*base.log_total_volume
folds = pd.read_csv(OUT / 'subject_folds.csv')
assert len(folds) == 278*10
assert not folds.duplicated(['subject_idx', 'repeat']).any()
assert archive['predictions'].shape == (4, 10, 278, 5, 2)
assert np.isfinite(archive['predictions']).all()
maximum_error = 0.
for repeat in range(10):
    assignment = folds.loc[folds['repeat'] == repeat].set_index('subject_idx').loc[base.subject_idx, 'fold'].to_numpy()
    assert set(assignment) == set(range(10))
    # All 278 subjects appear once per repeat, and each fold contains both sexes.
    assert len(assignment) == len(base)
    for fold in range(10):
        test = assignment == fold
        train = ~test
        assert set(base.loc[test, 'sex']) == {'F', 'M'}
        assert (base.loc[test, 'sex'] == 'F').sum() == 15
        assert (base.loc[test, 'sex'] == 'M').sum() in [12, 13]
        for m, model in enumerate(archive['model_names']):
            raw = sm.add_constant(base[protocol['models'][str(model)]].to_numpy(float), has_constant='add')
            # Independent regression library, unstandardized predictors.
            for j in range(5):
                for k in range(2):
                    y = archive['target'][:, j, k]
                    fit = sm.OLS(np.log(y[train]), raw[train]).fit()
                    predicted = np.exp(fit.predict(raw[test]))*np.exp(fit.resid).mean()
                    difference = np.max(abs(predicted-archive['predictions'][m, repeat, test, j, k]))
                    maximum_error = max(maximum_error, float(difference))
                    np.testing.assert_allclose(predicted, archive['predictions'][m, repeat, test, j, k], rtol=1e-9, atol=1e-10)
metrics = pd.read_csv(OUT / 'prediction_metrics.csv')
for row in metrics.itertuples():
    m = archive['model_names'].tolist().index(row.model)
    j = archive['loads'].tolist().index(row.load_case)
    k = archive['outcomes'].tolist().index(row.outcome)
    y = archive['target'][:, j, k]
    err = archive['predictions'][m, :, :, j, k]-y
    np.testing.assert_allclose([row.rmse, row.mae, row.cv_r2],
                               [np.sqrt(np.mean(err**2)), np.mean(abs(err)), 1-np.mean(err**2)/np.var(y)], atol=1e-12)
mixture = pd.read_csv(OUT / 'mixture_subject_endpoints.csv')
assert len(mixture) == 278*5 and not mixture.duplicated(['subject_idx', 'weight']).any()
endpoint_error = 0.
for i in range(278):
    with np.load(OUT / 'load_mixture_checkpoints' / f'subject_{i:03d}.npz') as check, np.load(PAPER / 'tables/corrected' / f'subject_{i:03d}.npz') as saved:
        assert np.isfinite(check['translations']).all() and np.isfinite(check['angles']).all()
        np.testing.assert_array_equal(check['weights'], [0, .25, .5, .75, 1])
        for index, load in [(0, 'LAB_phase1'), (-1, 'LAB_phase2')]:
            for key, field in [('translations', 'trans'), ('angles', 'angles')]:
                delta = abs(check[key][index]-saved[f'full_{load}_{field}']).max()
                endpoint_error = max(endpoint_error, float(delta))
                np.testing.assert_allclose(check[key][index], saved[f'full_{load}_{field}'], rtol=1e-8, atol=1e-8)
        # The vector estimate must not exceed its scalar counterpart.
        for row in mixture.loc[mixture.subject_idx == i].itertuples():
            assert row.vector_oracle_mm <= row.scalar_oracle_mm+1e-12
            w = row.weight
            d = (1-w)*check['translations'][0]+w*check['translations'][-1]
            np.testing.assert_allclose(np.linalg.norm(d, axis=1).mean(), row.vector_oracle_mm, atol=1e-12)
summary = pd.read_csv(OUT / 'mixture_summary.csv')
for row in summary.itertuples():
    sub = mixture.loc[mixture.weight == row.weight]
    np.testing.assert_allclose(row.scalar_oracle_rmse_mm, np.sqrt(np.mean((sub.scalar_oracle_mm-sub.translation_mm)**2)), atol=1e-12)
    np.testing.assert_allclose(row.vector_oracle_rmse_mm, np.sqrt(np.mean((sub.vector_oracle_mm-sub.translation_mm)**2)), atol=1e-12)
    np.testing.assert_allclose(row.median_translation_mm, sub.translation_mm.median(), atol=1e-12)
    np.testing.assert_allclose(row.median_directional_reduction_pct, sub.directional_reduction_pct.median(), atol=1e-12)
checks = dict(status='passed', held_out_regressions_independently_refitted=4000,
              max_arithmetic_prediction_difference=maximum_error,
              subjects_checked_for_both_mixture_endpoints=278,
              maximum_endpoint_component_difference=endpoint_error,
              checks=['paired subject folds and sex stratification', 'training-only raw-unit statsmodels refits',
                      'saved prediction metrics from pooled errors', 'all mixture endpoint components',
                      'vector triangle inequality and oracle definitions', 'mixed-field summary errors'])
(OUT / 'validation_checks.json').write_text(json.dumps(checks, indent=2))
print(json.dumps(checks, indent=2))
