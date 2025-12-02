# Changelog

## Unreleased

- Init strategy: Added configurable `init_strategy` option to FitConfig for MCMC initialisation. Choose from `"median"` (default), `"mean"`, or `"uniform"`.
- Train/test split: Added `test` parameter to `extract_features` processor for train/test data splitting. Standardisation uses train-only statistics, and category ordering is consistent across train and test sets.
- Dummy coding: Allow users to specify which parameter is the reference category and the full order of parameters when using dummy coding.
- Dummy coding: Fix name collision in `linear_group_binomial` when using dummy coding (changed sample site name to `{effect}_effects_raw`). Thank you Mitch Fruin for this!
- Dummy coding: Fix identifiability issue in `ordered_logistic_model` dummy coding (added reference category constraint and deterministic site). Thank you Mitch Fruin for this!
- Dummy coding: Fix same issues in custom models (`hierarchical_ordered_logistic_model`, `pairwise_logistic_model`). Thank you Mitch Fruin for this!
- Config validation: Warn users when config keys are misspelled (previously silently ignored invalid keys).
- Data loading: CSV and parquet loading now handled by loader instead of process, making it more intuitive and allowing users to use `hibayes-full`.
- Logging: Separate logs for each stage of the pipeline.
- Errors: More informative error messages when extracting samples.
- Docs: Updated documentation covering analysis state, models, platform, processors, and fit config.
- Bugfix: Fixed docs that incorrectly used `paths` instead of `path`.
