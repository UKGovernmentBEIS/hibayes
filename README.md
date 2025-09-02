# HiBayES

A Python package for analysing data from [Inspect](https://github.com/UKGovernmentBEIS/inspect_ai) logs using statistical modeling techniques presented in [HiBayES: A Hierarchical Bayesian Modeling Framework for AI Evaluation Statistics](https://arxiv.org/abs/2505.05602).

## 🚧 Work In Progress 🚧

This package is currently in development. Functionality will change and bugs are expected. **We very much value your feedback and contributions**. What level of automation is desired? Should we use Patsy for model configuration? etc.. Please open an issue or pull request if you have any suggestions or find any bugs.

There are a list of known issues listed in issues.

## Table of Contents
- [HiBayES](#hibayes)
  - [🚧 Work In Progress 🚧](#-work-in-progress-)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [DVC](#dvc)
  - [Quick start examples](#quick-start-examples)
  - [Features](#features)
  - [Command line usage](#example-command-line-usage)
    - [loading](#loading)
    - [modelling](#modelling)
    - [communicating](#communicating)
    - [Full pipeline](#full-pipeline)
  - [Configuration](#configuration)

## Installation

From source:
```bash
git clone git@github.com:UKGovernmentBEIS/hibayes.git
cd hibayes
pip install -e .
```

or with uv

```bash
git clone git@github.com:UKGovernmentBEIS/hibayes.git
cd hibayes
uv venv .venv
uv sync # if you want to exactly match dependencies
uv pip install -e .
```

or simply without git clone:

```bash
uv pip install git+https://github.com/UKGovernmentBEIS/hibayes.git
```

If you want to use a gpu for model fitting, you will need to install the optional gpu dependencies:

```bash
uv pip install "git+https://github.com/UKGovernmentBEIS/hibayes.git[gpu]"
```


if you would like to contribute please install the dev dependencies:

```bash
uv pip install -e .[dev]
```
We use pre-commit to ensure code quality. To set this up run:

```bash
uv run pre-commit install
```
This will set up pre-commit hooks to run on every commit. You can also run the pre-commit hooks manually with:

```bash
uv run pre-commit run --all-files
```
to then make a commit.
```bash
git add .
uv run commit -m "your commit message"
```
Note that you need to have your environment activated for the pre-commit hooks to work.

## DVC
This package uses [DVC](https://dvc.org/) for data version control. This ensures that all analysis is reproducible/auditable.

To reproduce analysis cd to an examples directory and run `dvc repro`.

```bash
cd examples/hibayes-usecase1/
uv run dvc repro
```
This will run the pipeline defined in the `dvc.yaml` file. This will run the full analysis pipeline. If you run it again, only the stages that have changed will run. Check the `dvc.yaml` file for details on which commands are run and what inputs they depend on/track.

To set up your own experiment and have DVC track it (assuming you have installed hibayes):

```bash
mkdir .experiments/name_of_experiment
cd .experiments/
git init # need git to track the dvc files
cd name_of_experiment
uv run dvc init --subdir # you can have multiple experiments in .experiments
```

OR if you want to contribute your experiment to the main repo as an example:

```bash
mkdir examples/name_of_experiment
cd examples/name_of_experiment
uv run dvc init --subdir
```

This will create a `.dvc` directory and a `.dvcignore` file. You can then create your DC pipeline by adding stages to the `dvc.yaml` file. See [example dvc.yaml](exmples/cybench/dvc.yaml) for a sample pipeline.

Then you can run the pipeline with:

```bash
dvc repro
```


## quick start examples.

_note: will be adding examples over the next few week. Current blocker is being able to share inspect eval logs run with AISI api keys. Noting that the gaia example is currently not working as requires access to private AISI s3.

```bash
cd examples/hibayes-usecase1/
uv run dvc repro
```
or without dvc
```bash
cd examples/hibayes-usecase1/
uv run hibayes-full --config files/config.yaml \
    --out .output
```

Fitting a statistical model to data is never plug and play and therefore there are a number of checks that are run to ensure the appropriateness of your model. Some require user approval e.g. prior predictive checks:

<img src="figs/prior-pred.png" alt="prior predictive check" width="500"/>



Checkout the output directory for the saved [analysis_state](src/hibayes/analysis_state.py) containing all the results of the analysis.

## Features

HiBayES consists of the following components which can be chained together to form a complete analysis pipeline:

1. **Loading** data from multiple Inspect AI logs with customisable extractors
    - this approach will be updated to match the recent introduction of [inspect dataframes](https://inspect.aisi.org.uk/dataframe.html).
2. **Modeling** data using statistical models
3. **Communicating** results with detailed diagnostics and visualisations

hibayes is designed so that any component can be run in isolation, allowing for flexiblity and the possibility for data/model exploration.

## Example command line usage

### loading
```bash
uv run hibayes-load --config <path-to-config.yaml> \
    --out <path-to-store-processed-data>
```
### modelling
```bash
uv run hibayes-model --config <path-to-config.yaml> \
    --data .output/load/data.parquet \
    --out <path-to-model-fit-results>
```
### communicating
```bash
uv run hibayes-comm --config <path-to-config.yaml> \
      --analysis_state <path-to-model-fit-results> \
      --out <path-to-communicate-results>
```
### Full pipeline
```bash
uv run hibayes-full --config <path-to-config.yaml> \
    --out <path-to-model-fit-results>
```
## Naming Conventions

HiBayES follows consistent naming conventions to ensure compatibility across models and clear communication between components. These conventions apply to:
- Configuration file keys
- Model parameter names
- Data column names
- Internal variable naming

### Core Patterns

#### Effects and Parameters
```python
# Fixed and random effects
f"{effect_name}_effects"
# Examples: age_effects, model_effects, group_effects

# Interaction terms
f"{effect_1}_{effect_2}_interaction"
# Examples: age_model_interaction, group_task_interaction

```

#### Data Structure References
```python
# Index mappings
f"{feature}_index"
# Examples: model_index, task_index, group_index

# Count variables
f"num_{feature}"
# Examples: num_models, num_tasks, num_groups
```
## Configuration

The package uses YAML configuration files to define the analysis pipeline. Here we detail some of the features you can use in th configuration file.

```yaml
data_loader:
  paths:
    files_to_process: # provide a list of files to process, these can be .json or .eval files
      - path/to/a/log.eval
      - path/to/logs/ # or a dir of logs
      - path/to/file/which/list/logs.txt or a txt file with a list of dirs/logs.
  extractors:
    enabled:
      - base # select from a set of extractors which specify what you would like extracted from the eval logs.
      - tokens
      - tools
    custom: # optional: add your own custom extractors
      path: path/to/custom_extractors.py
      classes:
        - CustomMetadataExtractor # list of custom extractors you would like you use see examples/gaia/files/extractors.py as an example.

model:
  models: # list of default models to run. See src/hibayes/model/model.py for available models.
    ModelOneName:
      column_map:
        score: success # optional mapping from data names to model feature names
    ModelTwoName:
      parameters:
      configurable_parameters: # optional: update model parameters with custom distributions and hyperparameters
        name: overall_mean # name of parameter - note: it must be listed as a 'configurable_parameter' in the model
        prior:
          distribution: normal
          distribution_args:
            loc: 0
            scale: 100 # here we increase the std of the normal prior distribution to 100 :O

checkers: # list of methods to check the model fit.
  checks: # these are all available checks
      - prior_predictive_plot
      - r_hat
      - divergences
      - ess_bulk
      - ess_tail
      - loo
      - bfmi
      - posterior_predictive_plot
      - waic
  custom_checks: # see examples/gaia/files/custom_checker.py for a mock example
    path: path/to/custom_checks.py
    checks:
      - posterior_mean_positive: {param: mu_overall, threshold: 0} # name of the checker with optional parameters

communicators:
  communicate:
    - forest_plot: {combined: true} #nupdate args
    - trace_plot
    - summary_table
  custom_communicators:
   - path: path/to/custom_communicators.py
     communicators:
       - custom_plot # name of the communicator with optional parameters
```


How to contribute your own custom checkers, models, communicators and extractors?

Check out the corresponding files in src/hibayes/{analyse, communicate, model, load} for examples. Below we detail how to implement your own checker.


```python
from hibayes.analyse import checker, CheckerResult, Checker

@checker(when="before") # run this check after the model is fitted
def custom_checker(message: str = "great job") -> Checker:
    """
    An example custom checker that does nothing that useful.
    """
    def check(state: ModelAnalysisState, display: ModellingDisplay) -> Tuple[ModelAnalysisState, CheckerResult]:
        """
        Run the check and return the state and result.
        """
        # do some checking
        if state.is_fitted:
            display.logger.info(message)
            return state, "pass"
        else:
            return state, "fail"
```

One aspect we are actively trying to improve is the model configuration. Only a small proportion of the model is fully configurable from the configs. It is not very flexible. We are considering moving to patsy however worry that it will not support a diverse enough set of models. If you have any suggestions please open an issue or pull request.
