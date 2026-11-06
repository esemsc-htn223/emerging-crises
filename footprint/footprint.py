import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The carbon footprint of an ML workflow

    This notebook trains the LSTM crisis classifier from `demos/5.LSTMClassifier.ipynb`
    in a simplified, single-domain form (developed markets only) and records the
    energy use and emissions of **every stage** as a timeseries.

    | ID | Stage |
    |---|---|
    | 0.x | code setup: idle baseline, imports, configuration |
    | 1.x | data loading and processing |
    | 2.x | hyperparameter optimisation (one ID per trial) |
    | 3.x | final training with the best hyperparameters |
    | 4.x | application: predicting on the held-out test set |

    **How it works.** A single CodeCarbon `EmissionsTracker` runs for the whole
    workflow and samples power every `MEASURE_POWER_SECS` seconds. A custom output
    handler (`stage_tracking.py`) writes one CSV row per sample, tagged with the
    current stage. `fp.stage(...)` closes the previous stage exactly at the call
    site. The training code in `src/` is not modified.

    Run headless with `pixi run python footprint/footprint.py`, or interactively
    with `pixi run marimo edit footprint/footprint.py`.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Configuration and tracker start

    Sampling interval, idle baseline and the HPO grid, plus the optional
    environment variables for comparing machines. The tracker starts at the
    end of this cell (stage 0.0, idle baseline), before the heavy imports.
    """)
    return


@app.cell
def _(mo):
    # ---- configuration ------------------------------------------------------
    MEASURE_POWER_SECS = 5      # sampling interval of the timeseries (seconds)
    IDLE_BASELINE_S = 30        # 0.0: measure an idle machine first (0 to skip)

    # Hyperparameter grid (stage 2). Each combination is one trial, ID 2.1 ... 2.N.
    # The number of trials is the main driver of total emissions.
    HPO_GRID = {
        "learning_rate": [8e-4, 3e-4],
        "lstm_units": [64, 128],
        "n_epochs": [10, 20],
    }

    import os

    if os.getenv("FOOTPRINT_SMOKE"):  # quick end-to-end check of the pipeline
        IDLE_BASELINE_S = 6
        HPO_GRID = {"learning_rate": [8e-4, 3e-4], "n_epochs": [1]}

    # For comparing machines (all optional, set as environment variables):
    #   FOOTPRINT_FINAL_TRIAL=8         final training (3.1) uses trial 2.8's params
    #                                   instead of this machine's HPO winner, so
    #                                   every machine trains the same final model
    #   FOOTPRINT_CARBON_INTENSITY=240  fixed grid intensity (gCO2e/kWh) instead of
    #                                   CodeCarbon's location lookup
    #   FOOTPRINT_PUE=1.2               datacentre power usage effectiveness (default 1)
    #   FOOTPRINT_TRACKING_MODE=process codecarbon's process mode (estimated CPU power
    #                                   from this process's load; disables PowerMetrics).
    #                                   Not needed for process power: every run also
    #                                   records this process's share of the machine's
    #                                   CPU time, see power="process" in plotting.py
    FINAL_TRIAL = int(os.getenv("FOOTPRINT_FINAL_TRIAL", "0")) or None
    _tracker_extra = {}
    if os.getenv("FOOTPRINT_CARBON_INTENSITY"):
        _tracker_extra["force_carbon_intensity_g_co2e_kwh"] = float(os.getenv("FOOTPRINT_CARBON_INTENSITY"))
    if os.getenv("FOOTPRINT_PUE"):
        _tracker_extra["pue"] = float(os.getenv("FOOTPRINT_PUE"))

    import sys
    import time
    from pathlib import Path

    NOTEBOOK_DIR = Path(mo.notebook_dir() or Path(__file__).parent)
    PROJECT_ROOT = NOTEBOOK_DIR.parent
    # smoke runs go to their own folder so they never mix with real results
    OUT_DIR = Path(os.getenv("FOOTPRINT_OUT") or NOTEBOOK_DIR / ("outputs_smoke" if os.getenv("FOOTPRINT_SMOKE") else "outputs"))
    if str(NOTEBOOK_DIR) not in sys.path:
        sys.path.insert(0, str(NOTEBOOK_DIR))

    import matplotlib

    if not mo.running_in_notebook():
        matplotlib.use("Agg")  # headless run: never block on plt.show()

    # ---- start tracking -----------------------------------------------------
    # NB: CodeCarbon's hardware detection happens inside StageTracker() and is
    # not itself measured.
    from stage_tracking import StageTracker

    fp = StageTracker(out_dir=OUT_DIR, measure_power_secs=MEASURE_POWER_SECS, **_tracker_extra)
    if IDLE_BASELINE_S > 0:
        fp.start("0.0", "idle-baseline")
        time.sleep(IDLE_BASELINE_S)
        fp.stage("0.1", "imports")
    else:
        fp.start("0.1", "imports")
    return FINAL_TRIAL, HPO_GRID, OUT_DIR, PROJECT_ROOT, fp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 0 · Setup: imports and configuration

    Stage 0.1 measures the library imports and 0.2 the seed, device and
    model parameters.
    """)
    return


@app.cell
def _(PROJECT_ROOT, fp):
    # ---- 0.1 imports (stage opened in the cell above) ----------------------
    import gc
    import itertools
    import json

    import os as _os

    import dotenv
    import numpy as np
    import pandas as pd
    import torch
    from datasets import load_dataset
    from sklearn.metrics import roc_auc_score

    from data.data_utilities import build_labels
    from features.utilities import load_or_train_model
    from model.architectures import LSTMClassifier
    from model.classifiers import plot_metrics
    from model.loss import PrecisionFocalLoss
    from model.utilities import set_device, set_seed

    # ---- 0.2 configuration --------------------------------------------------
    fp.stage("0.2", "config")
    seed = set_seed()
    device = set_device()
    dotenv.load_dotenv(PROJECT_ROOT / ".env")
    username = _os.getenv("HUGGINGFACE_USERNAME")

    # tuned parameters from demos/5.LSTMClassifier.ipynb; HPO overrides some of them
    BASE_PARAMS = {
        "sequence_length": 48,
        "lstm_units": 128,
        "num_lstm_layers": 3,
        "dense_units": [128, 64],
        "learning_rate": 8e-4,
        "batch_size": 64,
        "dropout_rate": 0.15,
        "n_epochs": 20,
        "use_attention": True,
        "use_residual": True,
        "Gaussian_noise": False,
        "noise_factor": 0.0,
        "criterion": PrecisionFocalLoss(alpha=0.1, gamma=2.0, precision_weight=2.25),
    }
    return (
        BASE_PARAMS,
        LSTMClassifier,
        build_labels,
        device,
        gc,
        itertools,
        json,
        load_dataset,
        load_or_train_model,
        pd,
        plot_metrics,
        roc_auc_score,
        seed,
        username,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Data loading and processing

    1.1 loads the developed-market data from Hugging Face, 1.2 builds the
    crisis labels and 1.3 prepends training context to the test set.
    """)
    return


@app.cell
def _(BASE_PARAMS, build_labels, fp, load_dataset, pd, username):
    # ---- 1.1 download data --------------------------------------------------
    fp.stage("1.1", "download-data")
    datalink = f"{username}/imputed-monthly"
    train_df = load_dataset(datalink, "developed", split="train").to_pandas()
    test_df = load_dataset(datalink, "developed", split="test").to_pandas()

    # ---- 1.2 build labels ---------------------------------------------------
    fp.stage("1.2", "build-labels")
    y_train = build_labels(train_df)
    y_test = build_labels(test_df)

    # ---- 1.3 test context ---------------------------------------------------
    # prepend the last sequence_length-1 months of training data for each
    # country so every test month has a full input sequence
    fp.stage("1.3", "build-test-context")
    _n_context = train_df["Country"].nunique() * (BASE_PARAMS["sequence_length"] - 1)
    _context_df = train_df.sort_values("Date")[-_n_context:]
    add_test_df = pd.concat([_context_df, test_df], ignore_index=True)
    y_test_extended = build_labels(add_test_df)

    print(
        f"train {train_df.shape} ({y_train.mean() * 100:.2f}% positive), "
        f"test {test_df.shape} ({y_test.mean() * 100:.2f}% positive)"
    )
    return add_test_df, train_df, y_test_extended, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Hyperparameter optimisation

    One stage ID per trial (2.1 … 2.N), each scored on validation AUC from
    `fit()`'s own 80/20 split; the test set is never used here.
    """)
    return


@app.cell
def _(
    BASE_PARAMS,
    FINAL_TRIAL,
    HPO_GRID,
    LSTMClassifier,
    OUT_DIR,
    device,
    fp,
    gc,
    itertools,
    json,
    pd,
    seed,
    train_df,
    y_train,
):
    # ---- 2.0 HPO setup ------------------------------------------------------
    fp.stage("2.0", "hpo-setup")
    _keys = list(HPO_GRID)
    trials = [dict(zip(_keys, _values)) for _values in itertools.product(*HPO_GRID.values())]
    print(f"HPO: {len(trials)} trials")

    # ---- 2.1 ... 2.N one stage per trial -------------------------------------
    # Scored on the validation AUC from fit()'s internal 80/20 split; the test
    # set is never seen during HPO.
    _results = []
    for _i, _trial in enumerate(trials, start=1):
        fp.stage(f"2.{_i}", "hpo-trial", **_trial)
        _model = LSTMClassifier(params={**BASE_PARAMS, **_trial}, seed=seed, device=device)
        _model.fit(train_df, y_train)
        _val_auc = _model.val_metrics_history["auc"][-1]
        _results.append({"stage_id": f"2.{_i}", **_trial, "val_auc": _val_auc})
        print(f"  2.{_i} {json.dumps(_trial)} -> val AUC {_val_auc:.4f}")
        del _model
        gc.collect()

    # ---- 2.(N+1) select best ------------------------------------------------
    fp.stage(f"2.{len(trials) + 1}", "hpo-select")
    hpo_results = pd.DataFrame(_results)
    # one file per run (first 8 chars of the run id) so reruns never overwrite
    hpo_results.to_csv(OUT_DIR / f"hpo_results_{fp.run_id[:8]}.csv", index=False)
    hpo_winner = int(hpo_results["val_auc"].idxmax()) + 1  # trial number, 2.<n>
    print(f"best: 2.{hpo_winner} {trials[hpo_winner - 1]}, val AUC {hpo_results['val_auc'].iloc[hpo_winner - 1]:.4f}")
    final_trial = FINAL_TRIAL or hpo_winner
    if FINAL_TRIAL:
        print(f"FOOTPRINT_FINAL_TRIAL set: final training uses trial 2.{final_trial} {trials[final_trial - 1]}")
    best_params = {**BASE_PARAMS, **trials[final_trial - 1]}
    return best_params, final_trial, hpo_results, hpo_winner, trials


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Final training

    Retrains with the winning hyperparameters (or `FOOTPRINT_FINAL_TRIAL`)
    and saves the model artifact.
    """)
    return


@app.cell
def _(LSTMClassifier, best_params, fp, load_or_train_model, train_df, y_train):
    # ---- 3.1 final training (retrain with best params + save artifact) ------
    fp.stage("3.1", "final-train")
    final_model = load_or_train_model(
        model_class=LSTMClassifier,
        params=best_params,
        train_data=train_df,
        train_labels=y_train,
        force_retrain=True,
    )
    return (final_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Application

    Predicts on the held-out test set and evaluates it, then stops the
    tracker.
    """)
    return


@app.cell
def _(
    add_test_df,
    final_model,
    fp,
    plot_metrics,
    roc_auc_score,
    y_test_extended,
):
    # ---- 4.1 predict on the held-out test set -------------------------------
    fp.stage("4.1", "predict")
    _pred, _proba, _ = final_model.predict_with_indices(add_test_df)
    _y_true = final_model.get_aligned_labels(add_test_df, y_test_extended)

    # ---- 4.2 evaluate -------------------------------------------------------
    fp.stage("4.2", "evaluate")
    test_auc = roc_auc_score(_y_true, _proba[:, 1])
    print(f"Test AUC: {test_auc:.4f}")
    plot_metrics(final_model, add_test_df, y_test_extended)

    # ---- end of tracked workflow --------------------------------------------
    fp.end()
    return (test_auc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where did the emissions go?

    Everything below runs after the tracker has stopped, so it is not counted.

    **Caveats for interpreting the numbers**
    - Check the `[footprint] power source:` line printed at start-up. On Apple
      Silicon with passwordless `sudo powermetrics` it reads `AppleSiliconChip`
      (measured CPU + GPU power; `stage_tracking.py` works around codecarbon 3.3.2
      never selecting it by default). Without that, CodeCarbon only *estimates*
      CPU power from TDP × CPU load, and that estimate came out roughly 5–10×
      higher than measured on an M4. RAM power is always a fixed per-GB estimate.
    - With PowerMetrics, each sample spends ~2.5 s running `powermetrics`.
      Out of the box, CodeCarbon never counts that time, so only ~50% of wall
      time gets any energy. `stage_tracking.py` fixes this by giving each
      component its own clock (`count_measurement_time=True`). Check the
      "energy coverage" in `analysis.py`: it should be ~100%. Each stage
      change still adds ~2.5 s to take that reading.
    - CodeCarbon's `cpu/gpu/ram_power` fields are run-to-date averages. The
      plots use per-interval energy ÷ duration instead (`plotting.py`).
    - The tracker is in *machine* mode: it measures the whole computer. Stage
      `0.0` (idle baseline) shows the background load you could subtract.
    - Carbon intensity is the national grid average for the detected country.
    - Data download (1.1) only counts this machine's energy. The network and
      the data centre serving the files are not included.
    """)
    return


@app.cell
def _(
    OUT_DIR,
    device,
    final_trial,
    fp,
    hpo_results,
    hpo_winner,
    json,
    mo,
    test_auc,
    trials,
):
    mo.stop(not fp.enabled, mo.md("Tracking disabled (`TRACK=0`): no emissions data."))
    import platform
    import socket

    import pandas as _pd
    import torch as _torch

    import plotting  # shared with analysis.py (read-only, no tracker)

    _ts = plotting.select_run(plotting.load_timeseries(fp.timeseries_path), fp.run_id)
    stage_summary = plotting.stage_summary(_ts)
    stage_summary.to_csv(OUT_DIR / f"emissions_by_stage_{fp.run_id[:8]}.csv", index=False)

    # conservation check: interval rows must add up to the run total
    print(
        f"sum of intervals {_ts['emissions_kg'].sum():.6e} kg vs run total "
        f"{_ts['cum_emissions_kg'].iloc[-1]:.6e} kg; test AUC {test_auc:.4f}; {len(hpo_results)} HPO trials"
    )

    fig = plotting.plot_overview(_ts)
    fig.savefig(OUT_DIR / f"emissions_by_stage_{fp.run_id[:8]}.png", dpi=150)

    # machine + settings metadata, for labelling and comparing runs
    _summary = _pd.read_csv(fp.summary_path)
    _summary = _summary[_summary["run_id"] == fp.run_id].iloc[-1]
    _energy = _ts["energy_kwh"].sum()
    run_info = {
        "run_id": fp.run_id,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_model": _summary["cpu_model"],
        "cpu_count": int(_summary["cpu_count"]),
        "gpu_model": None if _pd.isna(_summary["gpu_model"]) else _summary["gpu_model"],
        "ram_total_gb": float(_summary["ram_total_size"]),
        "torch_version": _torch.__version__,
        "torch_threads": _torch.get_num_threads(),
        "training_device": str(device),
        "power_source": fp.power_source,
        "tracking_mode": fp.tracking_mode,
        "count_measurement_time": fp.count_measurement_time,
        "country": _summary["country_name"],
        "carbon_intensity_g_per_kwh": 1000 * _ts["emissions_kg"].sum() / _energy if _energy else None,
        "pue": float(_summary["pue"]),
        "hpo_winner": f"2.{hpo_winner}",
        "final_trial": f"2.{final_trial}",
        "final_params": trials[final_trial - 1],
        "test_auc": float(test_auc),
        "duration_s": float(_ts["duration_s"].sum()),
        "energy_kwh": float(_energy),
        "emissions_kg": float(_ts["emissions_kg"].sum()),
        "median_energy_coverage": float(_ts["energy_coverage"].median()),
        "process_energy_kwh": float(_ts["proc_energy_kwh"].sum()) if plotting.has_process_data(_ts) else None,
        "process_share_of_energy": float(_ts["proc_energy_kwh"].sum() / _energy) if plotting.has_process_data(_ts) and _energy else None,
    }
    (OUT_DIR / f"run_info_{fp.run_id[:8]}.json").write_text(json.dumps(run_info, indent=2, default=str))
    print(json.dumps(run_info, indent=2, default=str))
    # explore further (without collecting data) in analysis.py
    stage_summary
    return



if __name__ == "__main__":
    app.run()
