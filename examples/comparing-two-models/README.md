# Comparing two models

Is gpt-4o better than Claude 3.5 Sonnet on a benchmark, and by how much? Three
steps, each adding one piece of structure:

| Step | What it adds |
|---|---|
| `step1-unpaired/` | Two accuracies on DS-1000, one per model |
| `step2-paired/` | A difficulty offset per item, shared by both models (both answered the same items), fitted alongside step 1's model for comparison |
| `step3-all-benchmarks/` | The paired model across boolq, RACE-H, MBPP and DS-1000, with a difference per benchmark |

Each step reports accuracy per model, the accuracy difference with a 95%
credible interval, and P(Claude better) in `.output/communicate/comparison_summary.csv`.

Run a step from its directory:

```bash
uv run hibayes-full --config files/config.yaml --out .output
```

The data are the Inspect logs in `../hibayes-usecases/data/`. All three steps
fit the built-in `accuracy_comparison` model with different switches
(`item_column: null` for step 1, `group_column: benchmark` for step 3),
summarise with the built-in `comparison_summary` communicator, and check the
fit with the built-in `agreement_check` checker. `custom_communicators.py`
adds three example-specific plots, registered through `communicate.path`:
what the priors imply (`prior_plot`), how often the two models agree
(`agreement_plot`), and the fitted spread of item difficulty
(`item_difficulty_plot`).

The walkthrough, including results and why each choice was made, is in the docs:
[`docs/examples/comparing-two-models.qmd`](../../docs/examples/comparing-two-models.qmd).
