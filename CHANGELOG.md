# Changelog

## Unreleased

- (2026-01-09) Frequent save: Added `frequent_save` parameter to `model()` and `communicate()` functions to save analysis state after each model fit or communicator run, preventing loss of work if processing crashes. The `out` parameter is now optional (only required when `frequent_save=True`). CLI commands (`hibayes-model`, `hibayes-full`) enable frequent saves by default; use `--no-frequent-save` to disable.
- (2025-12-02) Init strategy: Added configurable `init_strategy` option to FitConfig for MCMC initialisation. Choose from `"median"` (default), `"mean"`, or `"uniform"`.
- (2025-12-02) Train/test split: Added `test` parameter to `extract_features` processor for train/test data splitting. Standardisation uses train-only statistics, and category ordering is consistent across train and test sets.
- (2025-12-02) Dummy coding: Allow users to specify which parameter is the reference category and the full order of parameters when using dummy coding.
- (2025-12-02) Dummy coding: Fix name collision in `linear_group_binomial` when using dummy coding (changed sample site name to `{effect}_effects_raw`). Thank you Mitch Fruin for this!
- (2025-12-02) Dummy coding: Fix identifiability issue in `ordered_logistic_model` dummy coding (added reference category constraint and deterministic site). Thank you Mitch Fruin for this!
- (2025-12-02) Dummy coding: Fix same issues in custom models (`hierarchical_ordered_logistic_model`, `pairwise_logistic_model`). Thank you Mitch Fruin for this!
- (2025-12-02) Config validation: Warn users when config keys are misspelled (previously silently ignored invalid keys).
- (2025-12-02) Data loading: CSV and parquet loading now handled by loader instead of process, making it more intuitive and allowing users to use `hibayes-full`.
- (2025-12-02) Logging: Separate logs for each stage of the pipeline.
- (2025-12-02) Errors: More informative error messages when extracting samples.
- (2025-12-02) Docs: Updated documentation covering analysis state, models, platform, processors, and fit config.
- (2025-12-02) Bugfix: Fixed docs that incorrectly used `paths` instead of `path`.
