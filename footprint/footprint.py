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

    import sys
    import time
    from pathlib import Path

    NOTEBOOK_DIR = Path(mo.notebook_dir() or Path(__file__).parent)
    PROJECT_ROOT = NOTEBOOK_DIR.parent
    OUT_DIR = NOTEBOOK_DIR / "outputs"
    if str(NOTEBOOK_DIR) not in sys.path:
        sys.path.insert(0, str(NOTEBOOK_DIR))

    import matplotlib

    if not mo.running_in_notebook():
        matplotlib.use("Agg")  # headless run: never block on plt.show()

    # ---- start tracking -----------------------------------------------------
    # NB: CodeCarbon's hardware detection happens inside StageTracker() and is
    # not itself measured.
    from stage_tracking import StageTracker

    fp = StageTracker(out_dir=OUT_DIR, measure_power_secs=MEASURE_POWER_SECS)
    if IDLE_BASELINE_S > 0:
        fp.start("0.0", "idle-baseline")
        time.sleep(IDLE_BASELINE_S)
        fp.stage("0.1", "imports")
    else:
        fp.start("0.1", "imports")
    return HPO_GRID, OUT_DIR, PROJECT_ROOT, fp


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


@app.cell
def _(
    BASE_PARAMS,
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
    hpo_results.to_csv(OUT_DIR / "hpo_results.csv", index=False)
    _best_idx = int(hpo_results["val_auc"].idxmax())
    best_params = {**BASE_PARAMS, **trials[_best_idx]}
    print(f"best: {trials[_best_idx]}, val AUC {hpo_results['val_auc'].iloc[_best_idx]:.4f}")
    return best_params, hpo_results


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
    - PowerMetrics takes about a 1 s power snapshot per sample, which is scaled
      up to the whole 5 s interval, and each stage change adds about 2 s to
      take that reading.
    - The tracker is in *machine* mode: it measures the whole computer. Stage
      `0.0` (idle baseline) shows the background load you could subtract.
    - Carbon intensity is the national grid average for the detected country.
    - Data download (1.1) only counts this machine's energy. The network and
      the data centre serving the files are not included.
    """)
    return


@app.cell
def _(OUT_DIR, fp, hpo_results, mo, test_auc):
    mo.stop(not fp.enabled, mo.md("Tracking disabled (`TRACK=0`): no emissions data."))
    import matplotlib.pyplot as plt

    _ts = fp.timeseries()
    stage_summary = fp.summary()
    stage_summary.to_csv(OUT_DIR / "emissions_by_stage.csv", index=False)

    # conservation check: interval rows must add up to the run total
    _total = _ts["cum_emissions_kg"].iloc[-1]
    print(
        f"sum of intervals {_ts['emissions_kg'].sum():.6e} kg vs run total {_total:.6e} kg; "
        f"test AUC {test_auc:.4f}; {len(hpo_results)} HPO trials"
    )

    # fixed categorical order, one colour per major stage
    STAGE_COLOURS = {
        "0": ("#2a78d6", "0 setup"),
        "1": ("#eb6834", "1 data"),
        "2": ("#1baf7a", "2 HPO"),
        "3": ("#eda100", "3 final training"),
        "4": ("#e87ba4", "4 application"),
    }
    _ts["major"] = _ts["stage_id"].str.split(".").str[0]
    _ts["power_w"] = _ts["energy_kwh"] * 3.6e6 / _ts["duration_s"]
    _ts["t_start"] = _ts["elapsed_s"] - _ts["duration_s"]

    fig, (_ax_power, _ax_cum, _ax_bar) = plt.subplots(
        3, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [2, 2, 1.4]}
    )
    for _, _row in _ts.iterrows():
        _colour = STAGE_COLOURS.get(_row["major"], ("#888888", ""))[0]
        _ax_power.fill_between(
            [_row["t_start"], _row["elapsed_s"]], 0, _row["power_w"],
            step="pre", color=_colour, linewidth=0,
        )
    _ax_power.set_ylabel("Average power (W)")
    _ax_power.set_title("Power draw over the workflow, coloured by stage", loc="left")

    _ax_cum.plot(_ts["elapsed_s"], _ts["cum_emissions_kg"] * 1000, color="#52514e", lw=2)
    for _major, (_colour, _label) in STAGE_COLOURS.items():
        _seg = _ts[_ts["major"] == _major]
        if len(_seg):
            _ax_cum.axvspan(_seg["t_start"].min(), _seg["elapsed_s"].max(), color=_colour, alpha=0.18, lw=0)
    _ax_cum.set_ylabel("Cumulative emissions (g CO₂e)")
    _ax_cum.set_xlabel("Elapsed time (s)")

    _by_major = _ts.groupby("major")["emissions_kg"].sum() * 1000
    _labels = [STAGE_COLOURS[_m][1] for _m in _by_major.index]
    _bars = _ax_bar.barh(_labels, _by_major.values, color=[STAGE_COLOURS[_m][0] for _m in _by_major.index], height=0.6)
    _ax_bar.bar_label(_bars, labels=[f"{_v:.3f} g" for _v in _by_major.values], padding=4, color="#0b0b0b")
    _ax_bar.invert_yaxis()
    _ax_bar.set_xlabel("Emissions per stage (g CO₂e)")

    _handles = [plt.Rectangle((0, 0), 1, 1, color=_c) for _c, _ in STAGE_COLOURS.values()]
    fig.legend(_handles, [_l for _, _l in STAGE_COLOURS.values()], ncol=5, frameon=False, loc="upper center")
    for _ax in (_ax_power, _ax_cum, _ax_bar):
        _ax.spines[["top", "right"]].set_visible(False)
        _ax.grid(axis="x" if _ax is _ax_bar else "y", color="#e5e4df", lw=0.8)
        _ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_DIR / "emissions_by_stage.png", dpi=150)
    stage_summary
    return


if __name__ == "__main__":
    app.run()
