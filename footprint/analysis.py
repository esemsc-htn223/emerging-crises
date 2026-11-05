import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(r"""
    # Exploring the emissions results

    This notebook only **reads** the CSVs written by `footprint.py`. It never
    starts a CodeCarbon tracker, so opening and re-running it collects no new
    data. The plotting functions live in `plotting.py` and are shared with
    `footprint.py`.

    Open with `pixi run marimo edit footprint/analysis.py`.
    """)
    return (mo,)


@app.cell
def _(mo):
    import sys
    from pathlib import Path

    NOTEBOOK_DIR = Path(mo.notebook_dir() or Path(__file__).parent)
    if str(NOTEBOOK_DIR) not in sys.path:
        sys.path.insert(0, str(NOTEBOOK_DIR))

    import pandas as pd

    import plotting

    data_dir = mo.ui.text(value=str(NOTEBOOK_DIR / "outputs"), label="Outputs folder", full_width=True)
    data_dir
    return Path, data_dir, pd, plotting


@app.cell
def _(Path, data_dir, mo, pd, plotting):
    _dir = Path(data_dir.value)
    mo.stop(
        not (_dir / "emissions_timeseries.csv").exists(),
        mo.md(f"No `emissions_timeseries.csv` in `{_dir}`."),
    )
    ts_all = plotting.load_timeseries(_dir / "emissions_timeseries.csv")
    hpo_files = {_f.stem.removeprefix("hpo_results_"): _f for _f in _dir.glob("hpo_results_*.csv")}
    runs = plotting.list_runs(ts_all)
    mo.vstack([mo.md("### Runs in this file (longest first)"), mo.ui.table(runs, selection=None)])
    return hpo_files, runs, ts_all


@app.cell
def _(mo, runs, ts_all):
    _options = {
        f"{r.start:%d %b %H:%M} · {r.duration_min:.1f} min · {r.n_samples} samples · {r.run_id[:8]}": r.run_id
        for r in runs.itertuples()
    }
    run_picker = mo.ui.dropdown(options=_options, value=next(iter(_options)), label="Run")
    _stages = sorted(ts_all["stage_id"].unique(), key=lambda s: tuple(int(p) for p in s.split(".")))
    exclude = mo.ui.multiselect(options=_stages, value=[], label="Exclude stage IDs")
    time_unit = mo.ui.radio(options=["s", "min", "h"], value="min", label="Time axis", inline=True)
    smooth = mo.ui.slider(1, 24, value=1, label="Smoothing (samples)", show_value=True)
    y_max = mo.ui.number(start=0, stop=1000, step=1, value=0, label="Power y-max (0 = auto)")
    intensity = mo.ui.number(
        start=0, stop=2000, step=1, value=0, label="Carbon intensity override, gCO₂e/kWh (0 = as recorded)"
    )
    mo.vstack([run_picker, exclude, mo.hstack([time_unit, smooth, y_max], justify="start"), intensity])
    return exclude, intensity, run_picker, smooth, time_unit, y_max


@app.cell
def _(exclude, hpo_files, intensity, pd, plotting, run_picker, ts_all):
    ts = plotting.select_run(ts_all, run_picker.value)
    if intensity.value:
        # same energy, different grid: lets runs from different machines/places
        # be compared on one carbon intensity
        ts["emissions_kg"] = ts["energy_kwh"] * intensity.value / 1000
        ts["cum_emissions_kg"] = ts["emissions_kg"].cumsum()
    _hpo_path = hpo_files.get(run_picker.value[:8])
    hpo = pd.read_csv(_hpo_path, dtype={"stage_id": str}) if _hpo_path else None
    if exclude.value:
        # drop the excluded samples, then recompute cumulative emissions and a
        # continuous time axis so the remaining stages butt up against each other
        ts = ts[~ts["stage_id"].isin(exclude.value)].reset_index(drop=True)
        ts["elapsed_s"] = ts["duration_s"].cumsum()
        ts["t_start_s"] = ts["elapsed_s"] - ts["duration_s"]
        ts["cum_emissions_kg"] = ts["emissions_kg"].cumsum()
        ts["cum_energy_kwh"] = ts["energy_kwh"].cumsum()
    return hpo, ts


@app.cell(hide_code=True)
def _(mo, ts):
    _total_g = ts["emissions_kg"].sum() * 1000
    _energy_wh = ts["energy_kwh"].sum() * 1000
    _hours = ts["duration_s"].sum() / 3600
    _mean_w = ts["energy_kwh"].sum() * 3.6e6 / ts["duration_s"].sum()
    mo.hstack(
        [
            mo.stat(f"{_total_g:.2f} g", label="CO₂e emitted"),
            mo.stat(f"{_energy_wh:.2f} Wh", label="Energy"),
            mo.stat(f"{_hours:.2f} h", label="Tracked time"),
            mo.stat(f"{_mean_w:.1f} W", label="Mean power"),
            mo.stat(f"{ts['energy_coverage'].median():.0%}", label="Energy coverage (median)"),
        ],
        justify="start",
    )
    return


@app.cell
def _(plotting, smooth, time_unit, ts, y_max):
    # the plots from footprint.py
    plotting.plot_overview(ts, time_unit=time_unit.value, smooth=smooth.value, y_max=y_max.value or None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Presentation: reveal stages step by step

    List the sections in the order you want to reveal them. Use major stages
    (`3` = final training), single stage IDs (`2.4` = one HPO trial) or `all`
    (the whole run). For example, `3, all` gives a final-training slide and
    then a whole-run slide. The y-axes always stay fixed to the full run.
    *Time axis: selection* zooms each slide to what it shows. *Hidden* leaves
    unrevealed stages blank; *ghost* draws them in grey.
    """)
    return


@app.cell
def _(mo):
    reveal_order = mo.ui.text(value="3, all", label="Reveal order", full_width=True)
    reveal_mode = mo.ui.radio(options=["cumulative", "one at a time"], value="cumulative", label="Steps", inline=True)
    reveal_context = mo.ui.radio(options=["hidden", "ghost"], value="hidden", label="Unrevealed stages", inline=True)
    reveal_x_range = mo.ui.radio(options=["selection", "full"], value="selection", label="Time axis", inline=True)
    reveal_x_origin = mo.ui.radio(options=["run", "selection"], value="run", label="Time measured from start of", inline=True)
    mo.vstack(
        [
            reveal_order,
            mo.hstack([reveal_mode, reveal_context], justify="start"),
            mo.hstack([reveal_x_range, reveal_x_origin], justify="start"),
        ]
    )
    return reveal_context, reveal_mode, reveal_order, reveal_x_origin, reveal_x_range


@app.cell
def _(mo, plotting, reveal_mode, reveal_order):
    reveal_list = [_s.strip() for _s in reveal_order.value.split(",") if _s.strip()]
    reveal_cumulative = reveal_mode.value == "cumulative"
    reveal = plotting.reveal_steps(reveal_list, cumulative=reveal_cumulative)
    mo.stop(not reveal, mo.md("Enter at least one stage."))
    reveal_step = mo.ui.slider(1, len(reveal), value=1, label="Step", show_value=True)
    save_reveal = mo.ui.run_button(label="Save all steps as PNGs")
    mo.hstack([reveal_step, save_reveal], justify="start")
    return reveal, reveal_cumulative, reveal_list, reveal_step, save_reveal


@app.cell
def _(
    plotting,
    reveal,
    reveal_context,
    reveal_step,
    reveal_x_origin,
    reveal_x_range,
    smooth,
    time_unit,
    ts,
    y_max,
):
    _show = reveal[reveal_step.value - 1]
    plotting.plot_overview(
        ts,
        show=None if "all" in _show else _show,
        context=reveal_context.value,
        x_range=reveal_x_range.value,
        x_origin=reveal_x_origin.value,
        time_unit=time_unit.value,
        smooth=smooth.value,
        y_max=y_max.value or None,
    )
    return


@app.cell
def _(
    Path,
    data_dir,
    mo,
    plotting,
    reveal_context,
    reveal_cumulative,
    reveal_list,
    reveal_x_origin,
    reveal_x_range,
    save_reveal,
    smooth,
    time_unit,
    ts,
    y_max,
):
    mo.stop(not save_reveal.value)
    _paths = plotting.save_reveal_sequence(
        ts,
        order=reveal_list,
        cumulative=reveal_cumulative,
        out_dir=Path(data_dir.value) / "reveal",
        context=reveal_context.value,
        x_range=reveal_x_range.value,
        x_origin=reveal_x_origin.value,
        time_unit=time_unit.value,
        smooth=smooth.value,
        y_max=y_max.value or None,
    )
    mo.md("Saved:\n\n" + "\n".join(f"- `{_p}`" for _p in _paths))
    return


@app.cell
def _(mo, plotting, ts):
    mo.vstack(
        [
            mo.md("### Per stage"),
            mo.ui.table(plotting.major_summary(ts), selection=None),
            mo.ui.table(plotting.stage_summary(ts), selection=None),
        ]
    )
    return


@app.cell
def _(plotting, smooth, time_unit, ts):
    plotting.plot_power_components(ts, time_unit=time_unit.value, smooth=smooth.value)
    return


@app.cell(hide_code=True)
def _(mo, plotting, time_unit, ts):
    mo.vstack(
        [
            mo.md(r"""
    ### Energy coverage (data quality check)
    Out of the box, CodeCarbon multiplies each power reading by the time since
    the previous measurement *finished*. With Apple PowerMetrics each
    measurement itself takes ~2.5 s, and that time is never counted, so runs
    made before the `count_measurement_time` fix in `stage_tracking.py` cover
    only ~50% of wall time. RAM power is a fixed estimate, so RAM interval
    power ÷ RAM nominal power shows the covered fraction directly. 100% means
    everything was counted. Values well below 100% mean the totals above are
    underestimates. Spikes at stage boundaries are just the very short
    samples there.
    """),
            plotting.plot_energy_coverage(ts, time_unit=time_unit.value),
        ]
    )
    return


@app.cell
def _(hpo, mo, plotting, ts):
    mo.stop(hpo is None or not ts["stage"].eq("hpo-trial").any(), mo.md("No HPO trials (or no `hpo_results_<run>.csv`) for this run."))
    _fig, _trials = plotting.plot_hpo_cost(ts, hpo)
    mo.hstack([_fig, mo.ui.table(_trials, selection=None)], widths=[1, 1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Raw samples
    Filter, sort and transform the selected run's samples below. Use `ts`
    directly in a new cell for your own plots.
    """)
    return


@app.cell
def _(mo, ts):
    mo.ui.dataframe(ts)
    return


if __name__ == "__main__":
    app.run()
