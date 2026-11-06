import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exploring the emissions results

    This notebook only **reads** the CSVs written by `footprint.py`. It never
    starts a CodeCarbon tracker, so opening and re-running it collects no new
    data. The plotting functions live in `plotting.py` and are shared with
    `footprint.py`. Settings are plain Python values: edit them and run the
    cell.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import sys
    from pathlib import Path

    NOTEBOOK_DIR = Path(mo.notebook_dir() or Path(__file__).parent)
    if str(NOTEBOOK_DIR) not in sys.path:
        sys.path.insert(0, str(NOTEBOOK_DIR))

    import pandas as pd

    import plotting

    return NOTEBOOK_DIR, mo, pd, plotting


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Settings

    Which data to read, which run to explore, and the plot settings shared by
    every plot below (a plot's own arguments take precedence).

    - `RUN_ID`: the first few characters of a run ID from the table below;
      `None` picks the newest run.
    - `EXCLUDE_STAGES`: stage IDs to drop entirely, e.g. `["0.0"]` for the
      idle baseline; the remaining stages are joined up on the time axis.
    - `CARBON_INTENSITY`: gCO₂e/kWh to re-express the run's emissions on a
      fixed grid, for comparing machines; `None` keeps the recorded value.
    - `PLOT`: shared `plot_overview` arguments. `linewidth` sets the hatch
      lines, process outlines and cumulative lines together.
    """)
    return


@app.cell
def _(NOTEBOOK_DIR):
    DATA_DIR = NOTEBOOK_DIR / "outputs"
    RUN_ID = None
    EXCLUDE_STAGES = []
    CARBON_INTENSITY = None

    PLOT = dict(
        time_unit="min",
        smooth=1,
        y_max=None,
        process_style="hatch",
        linewidth=0.5,
        x_origin="selection",
        legend=True,
    )
    return CARBON_INTENSITY, DATA_DIR, EXCLUDE_STAGES, PLOT, RUN_ID


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data and runs

    Every run in the outputs folder's `emissions_timeseries.csv`, newest
    first.
    """)
    return


@app.cell
def _(DATA_DIR, plotting):
    if not (DATA_DIR / "emissions_timeseries.csv").exists():
        raise FileNotFoundError(f"No emissions_timeseries.csv in {DATA_DIR}")
    ts_all = plotting.load_timeseries(DATA_DIR / "emissions_timeseries.csv")
    hpo_files = {_f.stem.removeprefix("hpo_results_"): _f for _f in DATA_DIR.glob("hpo_results_*.csv")}
    runs = plotting.list_runs(ts_all).sort_values("start", ascending=False).reset_index(drop=True)
    runs
    return hpo_files, runs, ts_all


@app.cell
def _(
    CARBON_INTENSITY,
    EXCLUDE_STAGES,
    RUN_ID,
    hpo_files,
    pd,
    plotting,
    runs,
    ts_all,
):
    if RUN_ID is None:
        run_id = runs["run_id"].iloc[0]
    else:
        _matches = runs.loc[runs["run_id"].str.startswith(RUN_ID), "run_id"]
        if len(_matches) != 1:
            raise ValueError(f"RUN_ID {RUN_ID!r} matches {len(_matches)} runs; give more characters")
        run_id = _matches.iloc[0]

    ts = plotting.select_run(ts_all, run_id)
    if CARBON_INTENSITY:
        # same energy, different grid: lets runs from different machines/places
        # be compared on one carbon intensity
        ts = plotting.apply_carbon_intensity(ts, CARBON_INTENSITY)
    if EXCLUDE_STAGES:
        # drop the excluded samples, then recompute cumulative emissions and a
        # continuous time axis so the remaining stages butt up against each other
        ts = ts[~ts["stage_id"].isin(EXCLUDE_STAGES)].reset_index(drop=True)
        ts["elapsed_s"] = ts["duration_s"].cumsum()
        ts["t_start_s"] = ts["elapsed_s"] - ts["duration_s"]
        ts["cum_emissions_kg"] = ts["emissions_kg"].cumsum()
        ts["cum_energy_kwh"] = ts["energy_kwh"].cumsum()

    _hpo_path = hpo_files.get(run_id[:8])
    hpo = pd.read_csv(_hpo_path, dtype={"stage_id": str}) if _hpo_path else None
    has_process = plotting.has_process_data(ts)


    def view(power: str) -> str:
        """
        Power view to plot, falling back to machine power without process data.

        Parameters
        ----------
        power : {"machine", "process", "both"}
            Requested view.

        Returns
        -------
        str
            ``power``, or ``"machine"`` if the run has no process data.
        """
        return power if has_process else "machine"


    print(f"run {run_id}: {len(ts)} samples, process data: {has_process}")
    if not has_process:
        print("Recorded before process attribution was added: process views fall back to machine power.")
    return hpo, run_id, ts, view


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Overview

    Headline numbers for the selected run, then the overview plot.
    """)
    return


@app.cell
def _(mo, plotting, ts):
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
            *(
                [mo.stat(f"{ts['proc_energy_kwh'].sum() / ts['energy_kwh'].sum():.0%}", label="Process share of energy")]
                if plotting.has_process_data(ts)
                else []
            ),
        ],
        justify="start",
    )
    return


@app.cell
def _(PLOT, plotting, ts, view):
    plotting.plot_overview(
        ts,
        **{
            **PLOT,
            "show": "3",
            "context": "hidden",
            "x_range": "selection",
            "panels": ("power", "cumulative"),
            "power": view("process"),
            "legend": False,
        },
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Presentation slides

    Each entry in `SLIDES` is one slide: a dict of `plot_overview` arguments,
    on top of the shared `PLOT` settings. The main ones are:

    - `show`: `"3"` for final training, `"2.4"` for one HPO trial, `"all"` for the whole run.
    - `power`: `"machine"`, `"process"` or `"both"`. The process is always hatched and outlined, and in `"both"` it sits on top of the rest of the machine (solid).
    - `x_range`: `"selection"` zooms the time axis to what is shown; `"full"` keeps the whole run's time axis.
    - `y_scale`: `"shown"` fits the axes to what is drawn; `"run"` keeps the whole run's machine scale on every slide.
    - `panels`: any of `"power"`, `"cumulative"`, `"shares"`.

    `SLIDE` picks which slide to preview. Runs without process data draw
    `"process"`/`"both"` slides as machine power.
    """)
    return


@app.cell
def _(PLOT, mo, plotting, ts, view):
    SLIDES = [
        dict(show="3", power="process", x_range="selection", y_scale="shown", panels=["power"], legend=False),
        dict(show="3", power="both", x_range="selection", panels=["power"], legend=False),
        #dict(show=['3','4'], x_range='selection', power="both", panels=["power"]),
        dict(show="all", power="both", panels=["power"]),
        dict(show="all", power="both", panels=["power", "cumulative", "shares"]),
    ]

    # shared settings first, then each slide's own; process views fall back to
    # machine power for runs without process data
    slide_specs = [{**PLOT, **_slide, "power": view(_slide.get("power", "machine"))} for _slide in SLIDES]

    mo.accordion(
        {f'**Slide {_sl+1}**' : plotting.plot_overview(ts, **slide_specs[_sl]) for _sl in range(len(SLIDES))},
        multiple=True
    )

    return (slide_specs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Save the slides

    Set `SAVE = True` and run the cell to write one PNG per slide to
    `<DATA_DIR>/slides/`. Set it back to `False` afterwards, otherwise the
    files are rewritten whenever an upstream cell reruns.
    """)
    return


@app.cell
def _(DATA_DIR, plotting, run_id, slide_specs, ts):
    SAVE = False
    if SAVE:
        _paths = plotting.save_slides(ts, slide_specs, out_dir=DATA_DIR / "slides", prefix=f"slide_{run_id[:8]}")
        print("Saved:\n" + "\n".join(f"  {_p}" for _p in _paths))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Per-stage breakdown

    Totals per major stage (0–4), then per stage ID.
    """)
    return


@app.cell
def _(plotting, ts):
    plotting.major_summary(ts)
    return


@app.cell
def _(plotting, ts):
    plotting.stage_summary(ts)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Power by component

    Machine power split into CPU, GPU and RAM. RAM is a fixed estimate. When
    the run has process data, this process's attributed power is overlaid as
    a line.
    """)
    return


@app.cell
def _(PLOT, plotting, ts):
    plotting.plot_power_components(ts, time_unit=PLOT["time_unit"], smooth=PLOT["smooth"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Energy coverage (data quality check)

    Out of the box, CodeCarbon multiplies each power reading by the time since
    the previous measurement *finished*. With Apple PowerMetrics each
    measurement itself takes ~2.5 s, and that time is never counted, so runs
    made before the `count_measurement_time` fix in `stage_tracking.py` cover
    only ~50% of wall time. RAM power is a fixed estimate, so RAM interval
    power ÷ RAM nominal power shows the covered fraction directly. 100% means
    everything was counted. Values well below 100% mean the totals above are
    underestimates. Spikes at stage boundaries are just the very short
    samples there.
    """)
    return


@app.cell
def _(PLOT, plotting, ts):
    plotting.plot_energy_coverage(ts, time_unit=PLOT["time_unit"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## HPO trials: cost vs performance

    Emissions of each hyperparameter trial against its validation AUC. Set
    `HPO_POWER` to `"process"` for this process's emissions instead of the
    whole machine's.
    """)
    return


@app.cell
def _(hpo, mo, plotting, ts, view):
    HPO_POWER = "machine"
    if hpo is None or not ts["stage"].eq("hpo-trial").any():
        print("No HPO trials (or no hpo_results_<run>.csv) for this run.")
        _out = None
    else:
        _fig, _trials = plotting.plot_hpo_cost(ts, hpo, power=view(HPO_POWER))
        _out = mo.hstack([_fig, _trials], widths=[1, 1])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Raw samples

    The selected run's samples. Use `ts` directly in a new cell for your own
    plots.
    """)
    return


@app.cell
def _(ts):
    ts
    return


if __name__ == "__main__":
    app.run()
