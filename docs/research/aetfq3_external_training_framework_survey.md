# aetfq3-lab External Training Framework Survey

Date: 2026-06-06

Scope: Q3 Lab only. This document is for research planning and dependency triage. It does not train models, read market data, write `output/`, create `lab_advisory/`, connect Stable, connect QMT, or generate `OrderIntent`.

## Executive Decision

Recommended adoption order:

1. Optuna
2. MLflow
3. vectorbt
4. tsai
5. NeuralForecast
6. PyTorch Forecasting
7. Darts
8. Microsoft Qlib
9. FinRL / FinRL-X

Reasoning: start with experiment infrastructure that does not impose a market-data model, then add lightweight research/backtest tooling, then PyTorch time-series libraries, and keep full-stack quant/RL frameworks as reference material until Q3 Lab has stable adapters and data-contract tests.

## Fit Matrix

| Framework | A: false downgrade | E: same-sector ETF ranking | F: 5m K / PyTorch execution model | Direct Lab import? | Stable? | QMT? | Next smoke test? |
|---|---:|---:|---:|---|---|---|---|
| Optuna | High | High | High | Yes, as optimizer only | No | No | Yes |
| MLflow | Medium | Medium | Medium | Yes, tracking only | No | No | Yes |
| vectorbt | Low | High | Medium | Yes, backtest/research only | No | No | Yes |
| tsai | High | Medium | High | Yes, model research only | No | No | Yes |
| NeuralForecast | Medium | High | High | Yes, forecast research only | No | No | Yes |
| PyTorch Forecasting | Medium | Medium | High | Maybe, heavier deps | No | No | Yes, after tsai/NeuralForecast |
| Darts | Low-Medium | Medium | Medium | Maybe, wrapper-heavy | No | No | Maybe later |
| Microsoft Qlib | Medium | High | Medium | Not initially | No | No | Reference first |
| FinRL / FinRL-X | Low | Low-Medium | Medium as RL reference | Not initially | No | No | Reference first |

Definitions:

- High: clearly reusable for the research direction with small adapter cost.
- Medium: useful but requires target reformulation or adapter work.
- Low: mostly indirect value.
- Direct Lab import means an installable dependency inside Lab research tooling only, not Stable or QMT.

## Required Answers

### 1. Which fit A false downgrade?

Best fits:

- tsai: strongest direct fit for sequence classification/regression on per-symbol or event-window data. It is useful if A becomes "detect whether downgrade signal is false" from historical windows and engineered labels.
- Optuna: useful across any A model family for hyperparameter search, feature-window search, threshold calibration, and objective selection.
- MLflow: useful for A experiment tracking, run metadata, metrics, artifacts, and reproducibility.

Secondary fits:

- Microsoft Qlib: useful as a reference for factor pipelines, dataset handlers, and workflow separation. Direct import is too heavy initially because Qlib expects its own data conventions and full quant workflow.
- PyTorch Forecasting / NeuralForecast: useful if A is reframed as forecast-error, post-event return forecast, or counterfactual time-series forecasting rather than direct classification.
- Darts: useful for quick baselines and anomaly-detection style framing, but its `TimeSeries` abstraction may add adapter weight.

Poor direct fits:

- vectorbt: not a model-training framework. It can validate downstream strategy behavior after A emits signals.
- FinRL / FinRL-X: RL is not a natural first tool for false-downgrade classification.

### 2. Which fit E same-sector ETF ranking?

Best fits:

- vectorbt: strong for cross-sectional signal evaluation, vectorized portfolio simulation, parameter sweeps, and same-sector ranking rules once E produces rank scores.
- Microsoft Qlib: strong conceptual fit because Qlib is built around quantitative investment workflows, alpha models, datasets, and backtesting. It is a good reference for ranking/label contracts.
- NeuralForecast: useful when ETF ranking is driven by short-horizon forecasts, especially if many related series share model structure.
- Optuna and MLflow: useful for ranking objective tuning and experiment tracking.

Secondary fits:

- PyTorch Forecasting: useful for TFT/DeepAR/N-BEATS style forecasting if E needs covariate-rich forecasting before ranking.
- Darts: useful for baseline forecasting/backtesting comparisons, but less ideal if the Lab wants to preserve its own data contracts.
- tsai: useful if ranking is implemented from classification/regression models over windowed features.

Poor direct fits:

- FinRL / FinRL-X: possible only if E becomes portfolio-allocation RL. That is later-stage research, not the first ranking implementation.

### 3. Which fit F 5-minute K / PyTorch execution model?

Best fits:

- tsai: good for 5-minute K sequence classification/regression, windowed CNN/RNN/Transformer-style models, and PyTorch-first experiments.
- NeuralForecast: good for deep learning forecasts over many 5-minute series, with a relatively direct PyTorch forecasting focus.
- PyTorch Forecasting: good for covariate-rich 5-minute forecasting and interpretable models such as TFT, but dependency and data-loader complexity are higher.
- Optuna and MLflow: useful for tuning and tracking model versions, metrics, window sizes, and loss variants.

Secondary fits:

- vectorbt: useful to replay model outputs through vectorized execution/backtest assumptions. It should not become the execution engine.
- Darts: useful for quick model baselines, but its wrapper model can obscure lower-level PyTorch control.
- Microsoft Qlib: useful as a reference for high-frequency workflow organization, not as the first execution-model dependency.

Reference-only:

- FinRL / FinRL-X: useful for RL execution-environment concepts and separation of agent/environment/reward, but direct use risks importing broker assumptions and reward semantics that do not match Q3 Lab.

### 4. Which are only for borrowing ideas, not direct introduction?

Reference first:

- Microsoft Qlib: borrow dataset/workflow separation, experiment recorder ideas, alpha-model/backtest boundaries, and handler-style abstractions. Do not initially adopt its data store or full workflow.
- FinRL / FinRL-X: borrow RL environment vocabulary, state/action/reward separation, and benchmark discipline. Do not initially adopt it as an execution or broker layer.
- Darts: borrow unified baseline interfaces and backtesting ergonomics if needed. Avoid making Q3 Lab data flow depend on its `TimeSeries` container early.

Conditional direct import after smoke:

- PyTorch Forecasting: direct import only if NeuralForecast/tsai do not cover F needs or if TFT-style covariate handling becomes important.

### 5. Which can be installed for next-stage smoke tests?

First smoke group:

- Optuna: run an in-memory or SQLite-backed study on synthetic arrays only.
- MLflow: run local file-store tracking in a temporary Lab test directory, not `output/`.
- vectorbt: run synthetic price/signal arrays and confirm vectorized portfolio metrics.
- tsai: import, instantiate a tiny model or run a dataloader on synthetic tensors.
- NeuralForecast: import and run a tiny synthetic forecasting fit only if the next-stage task explicitly allows training; otherwise import and construct model objects only.

Second smoke group:

- PyTorch Forecasting: import, construct `TimeSeriesDataSet` from synthetic data, and instantiate a small model. Full fit should wait for an explicit training task.
- Darts: import, create synthetic `TimeSeries`, and instantiate baseline models.

Reference-only smoke, not priority:

- Microsoft Qlib: inspect install/import feasibility and data-handler docs. Avoid creating Qlib data directories in this repo.
- FinRL / FinRL-X: inspect package layout and example contracts only. Avoid broker, live-trading, or data-download examples.

### 6. Which cannot connect to Stable?

Current answer: all of them cannot connect to Stable in this task or the next dependency-smoke stage.

Reason: these are external research/training/backtest/experiment frameworks. Stable should only receive carefully reviewed, deterministic, minimal artifacts through an explicit adapter and promotion process. No direct dependency, callback, model registry, backtest result, or experiment state from these frameworks should be imported into Stable.

Especially blocked from Stable direct connection:

- FinRL / FinRL-X: RL policy/execution assumptions and live-trading integrations are too risky.
- Microsoft Qlib: full workflow/data-store coupling is too broad.
- MLflow: model registry and artifact promotion must not become an implicit Stable deployment path.
- vectorbt: backtest metrics must not be treated as production signals.

### 7. Which cannot connect to QMT?

Current answer: all of them cannot connect to QMT.

Reason: QMT is execution-adjacent. These frameworks may generate research signals, models, backtests, or experiment metadata, but none should send orders, create `OrderIntent`, call broker APIs, or own runtime execution. Any future QMT integration must be a separate, explicit, audited adapter outside this survey.

Especially blocked from QMT direct connection:

- FinRL / FinRL-X: contains trading-agent and execution-environment patterns that could be mistaken for broker integration.
- vectorbt: portfolio simulation is not execution.
- Qlib: backtest/execution abstractions are not QMT adapters.
- MLflow: model registry state must not trigger trading actions.

### 8. Recommended adoption sequence

1. Add lab-only experiment infrastructure: Optuna, then MLflow.
2. Add synthetic-data backtest smoke: vectorbt.
3. Add A/F model smoke: tsai.
4. Add F/E forecasting smoke: NeuralForecast.
5. Add covariate-rich F forecasting candidate: PyTorch Forecasting.
6. Add optional baseline wrapper only if useful: Darts.
7. Keep Qlib as design reference; revisit direct import only after data-contract tests exist.
8. Keep FinRL / FinRL-X as RL architecture reference; revisit only if Q3 Lab opens an explicit RL sandbox.

## Project Notes

### Microsoft Qlib

Qlib is a broad AI-oriented quantitative investment platform. It is valuable as a design reference for data handling, model workflow, backtesting boundaries, and experiment process. For Q3 Lab, its strongest relevance is E ranking and factor-style research. It is less attractive as an initial dependency because adopting Qlib tends to pull in its data format, handler conventions, recorder/workflow shape, and backtest stack.

Recommendation: reference first, no direct install in phase 1.

### vectorbt

vectorbt is useful for fast vectorized backtesting and parameter sweeps. It is not a training framework, and it should not be confused with execution. For Q3 Lab, it is attractive because synthetic signal arrays can validate ranking/backtest assumptions without reading real行情 or connecting QMT.

Recommendation: install smoke early for synthetic E/F strategy replay.

### FinRL / FinRL-X

FinRL is a DRL-for-finance framework, while FinRL-X is positioned around multi-agent and live-trading oriented workflows. The concepts are relevant for future RL experiments, but the current A/E/F directions do not need direct RL infrastructure. Direct adoption would risk importing broker assumptions, reward definitions, live trading examples, and environment semantics that are not aligned with the Lab boundary.

Recommendation: reference only until there is a dedicated RL sandbox.

### tsai

tsai is a PyTorch/fastai-oriented time-series AI library. It is a strong candidate for A and F because both can be expressed as windowed time-series classification/regression problems. It is also useful for quick baseline models before the Lab writes custom PyTorch model code.

Recommendation: install smoke early with synthetic tensors only.

### PyTorch Forecasting

PyTorch Forecasting is useful for covariate-rich forecasting models, especially if F needs TFT-style modeling or if E ranking depends on multi-horizon forecasts. It brings more dependency and data-loader structure than tsai or NeuralForecast, so it should follow lighter smoke tests.

Recommendation: second-wave smoke after tsai and NeuralForecast.

### NeuralForecast

NeuralForecast provides PyTorch deep learning forecasting models and is a strong candidate for F 5-minute series and E ranking by predicted return/relative strength. It is more forecasting-specific than tsai, so it is less direct for A classification unless A is reframed as forecast/realization mismatch.

Recommendation: install smoke early after infrastructure.

### Darts

Darts gives a unified time-series interface, many forecasting models, backtesting utilities, and anomaly-detection style tools. It is useful for baselines and comparing methods. The main concern is adapter gravity: the Lab may start conforming to Darts' `TimeSeries` representation instead of preserving its own contracts.

Recommendation: optional second-wave smoke; borrow API ideas first.

### Optuna

Optuna is the clearest low-risk dependency. It can optimize feature windows, model hyperparameters, rank-objective weights, thresholds, and backtest parameters across A/E/F without imposing a trading/data framework.

Recommendation: first dependency smoke.

### MLflow

MLflow is useful for run tracking, metrics, parameters, local artifacts, and model lifecycle records. In Q3 Lab it should be tracking-only at first. The model registry/deployment path must not be allowed to bridge into Stable.

Recommendation: first dependency smoke after Optuna, with local-only tracking and explicit directory controls.

## Suggested Smoke-Test Rules

- Use synthetic arrays or synthetic pandas data only.
- Do not read market data.
- Do not train unless a later task explicitly allows training.
- Do not write `output/`.
- Do not create `lab_advisory/`.
- Do not connect Stable or QMT.
- Do not create `OrderIntent`.
- Keep smoke code under a future Lab-only test path, not production runtime.

## Sources

- Microsoft Qlib: https://github.com/microsoft/qlib
- vectorbt: https://vectorbt.dev/
- FinRL: https://github.com/AI4Finance-Foundation/FinRL
- FinRL-X / FinRL-Trading: https://github.com/AI4Finance-Foundation/FinRL_Trading
- tsai: https://timeseriesai.github.io/tsai/
- PyTorch Forecasting: https://pytorch-forecasting.readthedocs.io/
- NeuralForecast: https://nixtlaverse.nixtla.io/neuralforecast/
- Darts: https://unit8co.github.io/darts/
- Optuna: https://optuna.org/
- MLflow: https://mlflow.org/docs/latest/
