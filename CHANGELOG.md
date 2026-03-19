# Changelog


- (2026-03-19) Textual TUI: Added interactive terminal UI with per-model dashboard, keyboard navigation, and display protocol abstraction (`--no-tui` falls back to Rich). Also fixed GPU/CPU platform initialisation and cleaned up progress bar handling.
- (2026-03-18) Bugfix: Fixed `ModelAnalysisState.load()` not restoring DataFrame/Dataset diagnostics (e.g. summary) from saved CSV/NC files, causing `get_summary()` to return a string after reload.
- (2026-03-18) Refactor: Replaced custom `LogProcessor` loading pipeline with `samples_df()` and bulk log reads. This removes `LogProcessor` class and `patch_inspect_loader.py`.
- (2026-01-09) Frequent save: Added `frequent_save` parameter to `model()` and `communicate()` functions to save analysis state after each model fit or communicator run, preventing loss of work if processing crashes. The `out` parameter is now optional (only required when `frequent_save=True`). CLI commands (`hibayes-model`, `hibayes-full`) enable frequent saves by default; use `--no-frequent-save` to disable.
- (2025-12-02) Major update: Added configurable init strategy, train/test splitting, and dummy coding improvements (thank you Mitch Fruin!). Also added config key validation, moved CSV/parquet loading to the loader stage, per-stage logging, better error messages, and updated docs.
