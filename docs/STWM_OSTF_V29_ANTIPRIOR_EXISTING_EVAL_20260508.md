# STWM OSTF V29 Anti-Prior Existing Eval

- manifest_dir: `manifests/ostf_v29_antiprior`
- prior_combos: `['M128_H32', 'M512_H32', 'M128_H64', 'M512_H64']`
- existing_v28_models: `['V28_H32_best_available', 'V28_H64_best_available', 'V28_M512_H32_available', 'V28_H64_wo_dense_points', 'V28_H64_wo_semantic_memory', 'V28_H64_wo_residual_modes']`
- bootstrap_path: `reports/stwm_ostf_v29_antiprior_existing_bootstrap_20260508.json`
- metric_schema_note: `Analytic priors include MissRate@128, endpoint threshold-AUC, PCK@64 and relative layout error. Existing V28 item-score reports do not store raw predictions, so V28 PCK@64/layout error are null; V28 MissRate@128 and threshold-AUC are derived from stored minFDE.`
