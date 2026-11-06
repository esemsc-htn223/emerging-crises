"""
Loading and plotting helpers for the stage-tagged emissions timeseries.

Read-only: nothing here imports codecarbon or starts a tracker, so it is safe
to use for exploring results without collecting new data.

Power columns
-------------
codecarbon's ``cpu_power_w`` / ``gpu_power_w`` / ``ram_power_w`` are *run-to-date
averages* (sum of readings / number of readings), not the power in each
interval, so they ramp smoothly and are unsuitable for a timeseries.

:func:`load_timeseries` adds true per-interval power from the interval
energies: ``cpu_w``, ``gpu_w``, ``ram_w`` and ``power_w`` (total) = energy /
duration. Very short samples (< ``MIN_SAMPLE_S``) at stage boundaries can carry
energy belonging to the neighbouring interval (codecarbon's scheduler thread and
``flush()`` race), which would show as huge spikes, so they are merged into the
preceding sample for the power calculation. Energy/emission totals are unchanged.

``energy_coverage`` = RAM interval power / RAM nominal power. RAM is a constant
estimate, so this is the fraction of wall time codecarbon actually counted
energy for: ~1 normally, ~0.5 for Apple PowerMetrics runs recorded before the
``count_measurement_time`` fix in ``stage_tracking.py``.

Process power
-------------
Runs recorded with the process-attribution columns also get the share of the
machine's power attributable to this Python process:

- ``proc_cpu_share``: process CPU seconds / machine busy CPU seconds in the
  interval, so ``proc_cpu_w = cpu_w * proc_cpu_share``.
- ``proc_ram_w = ram_w * proc_rss_gb / ram_total_gb``.
- GPU power is not attributed (the training runs on the CPU).
- ``proc_power_w``, ``proc_energy_kwh`` and ``proc_emissions_kg`` follow.

Machine minus process is the cost of everything else running at the time.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Patch

# one colour per major stage, fixed order (never cycled)
STAGE_COLOURS: dict[str, tuple[str, str]] = {
    "0": ("#2a78d6", "Code setup"),
    "1": ("#eb6834", "Data loading"),
    "2": ("#1baf7a", "Hyperparameter opt."),
    "3": ("#eda100", "Final training"),
    "4": ("#e87ba4", "Model application"),
}
COMPONENT_COLOURS: dict[str, str] = {"cpu": "#2a78d6", "gpu": "#eb6834", "ram": "#1baf7a"}
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e5e4df"
GHOST = "#d9d8d3"          # colour of unselected stages in context="ghost"
REST_ALPHA = 0.35          # opacity of the "rest of machine" layer (process_style="layers")
HATCH_COLOUR = "#2b2a28"   # hatch lines marking the process's share (process_style="hatch")
LINEWIDTH = 0.5            # default for hatch lines, process outlines and cumulative lines
CUMULATIVE_ALPHA = 0.18    # opacity of the stage colour under the cumulative-emissions line
TIME_UNITS: dict[str, float] = {"s": 1.0, "min": 60.0, "h": 3600.0}
MIN_SAMPLE_S = 0.5
PANEL_HEIGHTS: dict[str, float] = {"power": 2.0, "cumulative": 2.0, "shares": 1.4}
MIN_LEFT_MARGIN_IN = 0.95  # fixed minimum left margin, so the plot area doesn't shift with tick-label width

Show = str | Sequence[str] | None
"""Stage selection: None (all), a stage such as ``"3"`` or ``"2.4"``, or a list of them."""
PowerView = Literal["machine", "process", "both"]
ProcessStyle = Literal["hatch", "layers"]
Panel = Literal["power", "cumulative", "shares"]
TimeUnit = Literal["s", "min", "h"]


# ---- data -------------------------------------------------------------------
def load_timeseries(path: str | Path) -> pd.DataFrame:
    """
    Read ``emissions_timeseries.csv`` (all runs) and add derived columns.

    Parameters
    ----------
    path : str or Path
        Timeseries CSV written by ``stage_tracking.StageTracker``.

    Returns
    -------
    pandas.DataFrame
        The raw columns plus ``major``, ``t_start_s``, per-interval power
        (``cpu_w``, ``gpu_w``, ``ram_w``, ``power_w``), ``energy_coverage`` and,
        where recorded, the process columns described in the module docstring
        (NaN for runs recorded without them).
    """
    df = pd.read_csv(path, dtype={"stage_id": str, "run_id": str})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["major"] = df["stage_id"].str.split(".").str[0]
    df["t_start_s"] = df["elapsed_s"] - df["duration_s"]

    # per-interval power; short samples join the preceding sample of the same run
    block = (df["duration_s"] >= MIN_SAMPLE_S).groupby(df["run_id"]).cumsum()
    keys = [df["run_id"], block]
    grp = df.groupby(keys)
    dur = grp["duration_s"].transform("sum")
    for name, col in (("cpu_w", "cpu_energy_kwh"), ("gpu_w", "gpu_energy_kwh"), ("ram_w", "ram_energy_kwh"), ("power_w", "energy_kwh")):
        df[name] = (grp[col].transform("sum") * 3.6e6 / dur).where(dur > 0, 0.0)

    ram_nominal = df.groupby("run_id")["ram_power_w"].transform("median")
    df["energy_coverage"] = df["ram_w"] / ram_nominal

    if {"proc_cpu_s", "sys_busy_cpu_s", "proc_rss_gb", "ram_total_gb"}.issubset(df.columns):
        by_run = df.groupby("run_id")
        d_proc = by_run["proc_cpu_s"].diff()
        d_sys = by_run["sys_busy_cpu_s"].diff()
        blk_proc = d_proc.groupby(keys).transform(lambda s: s.sum(min_count=1))
        blk_sys = d_sys.groupby(keys).transform(lambda s: s.sum(min_count=1))
        share = (blk_proc / blk_sys.where(blk_sys > 0)).clip(0, 1)
        # the first sample of a run has no previous counter: borrow the next share
        df["proc_cpu_share"] = share.groupby(df["run_id"]).bfill()
        df["proc_ram_share"] = (df["proc_rss_gb"] / df["ram_total_gb"]).clip(0, 1)
        df["proc_cpu_w"] = df["cpu_w"] * df["proc_cpu_share"]
        df["proc_ram_w"] = df["ram_w"] * df["proc_ram_share"]
        df["proc_power_w"] = df["proc_cpu_w"] + df["proc_ram_w"]
        df["proc_energy_kwh"] = df["proc_power_w"] * df["duration_s"] / 3.6e6
        intensity = by_run["emissions_kg"].transform("sum") / by_run["energy_kwh"].transform("sum")
        df["proc_emissions_kg"] = df["proc_energy_kwh"] * intensity
    return df


def has_process_data(ts: pd.DataFrame) -> bool:
    """
    Whether a run carries the process-attribution columns.

    Parameters
    ----------
    ts : pandas.DataFrame
        Output of :func:`load_timeseries` (one or more runs).

    Returns
    -------
    bool
        True if any sample has a process power value.
    """
    return "proc_power_w" in ts.columns and bool(ts["proc_power_w"].notna().any())


def apply_carbon_intensity(ts: pd.DataFrame, g_per_kwh: float) -> pd.DataFrame:
    """
    Recompute emissions from energy with a fixed grid carbon intensity.

    Lets runs from different machines or locations be compared on one grid.

    Parameters
    ----------
    ts : pandas.DataFrame
        Output of :func:`load_timeseries`.
    g_per_kwh : float
        Carbon intensity in gCO2e/kWh.

    Returns
    -------
    pandas.DataFrame
        A copy with ``emissions_kg``, ``cum_emissions_kg`` and (if present)
        ``proc_emissions_kg`` recomputed.
    """
    ts = ts.copy()
    ts["emissions_kg"] = ts["energy_kwh"] * g_per_kwh / 1000
    ts["cum_emissions_kg"] = ts.groupby("run_id")["emissions_kg"].cumsum()
    if "proc_energy_kwh" in ts.columns:
        ts["proc_emissions_kg"] = ts["proc_energy_kwh"] * g_per_kwh / 1000
    return ts


def list_runs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise the runs in a timeseries, longest first.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`load_timeseries`.

    Returns
    -------
    pandas.DataFrame
        One row per ``run_id`` with start/end time, duration, sample count,
        stage range, total emissions, tracking mode and whether process data
        was recorded.
    """
    runs = (
        df.groupby("run_id", sort=False)
        .agg(
            start=("timestamp", "first"),
            end=("timestamp", "last"),
            duration_min=("elapsed_s", lambda s: s.max() / 60),
            n_samples=("elapsed_s", "size"),
            stages=("stage_id", lambda s: f"{s.iloc[0]} → {s.iloc[-1]} ({s.nunique()} ids)"),
            emissions_g=("emissions_kg", lambda s: s.sum() * 1000),
        )
        .reset_index()
    )
    by_run = df.groupby("run_id", sort=False)
    runs["tracking_mode"] = (
        by_run["tracking_mode"].first().fillna("machine").values if "tracking_mode" in df.columns else "machine"
    )
    runs["process_data"] = (
        by_run["proc_power_w"].apply(lambda s: s.notna().any()).values if "proc_power_w" in df.columns else False
    )
    return runs.sort_values("duration_min", ascending=False).reset_index(drop=True)


def select_run(df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    """
    Rows of one run.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`load_timeseries`.
    run_id : str
        Run to keep.

    Returns
    -------
    pandas.DataFrame
        That run's rows with a fresh index.
    """
    return df[df["run_id"] == run_id].reset_index(drop=True)


def stage_summary(ts: pd.DataFrame) -> pd.DataFrame:
    """
    Energy, emissions and duration per stage ID.

    Parameters
    ----------
    ts : pandas.DataFrame
        One run from :func:`load_timeseries`.

    Returns
    -------
    pandas.DataFrame
        One row per stage ID, with mean machine and CPU power, median energy
        coverage, share of the run's emissions and, when recorded, the
        process's energy, emissions and share of machine energy.
    """
    aggs = {
        "major_stage": ("major", "first"),
        "duration_s": ("duration_s", "sum"),
        "n_samples": ("duration_s", "size"),
        "energy_kwh": ("energy_kwh", "sum"),
        "emissions_kg": ("emissions_kg", "sum"),
        "cpu_energy_kwh": ("cpu_energy_kwh", "sum"),
        "energy_coverage": ("energy_coverage", "median"),
    }
    process = has_process_data(ts)
    if process:
        aggs |= {"proc_energy_kwh": ("proc_energy_kwh", "sum"), "proc_emissions_kg": ("proc_emissions_kg", "sum")}
    out = ts.groupby(["stage_id", "stage"], sort=False).agg(**aggs).reset_index()
    out["mean_power_w"] = out["energy_kwh"] * 3.6e6 / out["duration_s"]
    out["mean_cpu_power_w"] = out.pop("cpu_energy_kwh") * 3.6e6 / out["duration_s"]
    out["share_of_emissions_pct"] = 100 * out["emissions_kg"] / out["emissions_kg"].sum()
    if process:
        out["process_share_of_energy_pct"] = 100 * out["proc_energy_kwh"] / out["energy_kwh"]
    return out


def major_summary(ts: pd.DataFrame) -> pd.DataFrame:
    """
    Duration, energy and emissions per major stage (0-4).

    Parameters
    ----------
    ts : pandas.DataFrame
        One run from :func:`load_timeseries`.

    Returns
    -------
    pandas.DataFrame
        One row per major stage with its share of the run's emissions and,
        when recorded, the process's energy and emissions.
    """
    aggs = {"duration_s": ("duration_s", "sum"), "energy_kwh": ("energy_kwh", "sum"), "emissions_kg": ("emissions_kg", "sum")}
    if has_process_data(ts):
        aggs |= {"proc_energy_kwh": ("proc_energy_kwh", "sum"), "proc_emissions_kg": ("proc_emissions_kg", "sum")}
    out = ts.groupby("major").agg(**aggs).reset_index()
    out.insert(1, "stage", out["major"].map(lambda m: STAGE_COLOURS.get(m, ("", m))[1]))
    out["share_of_emissions_pct"] = 100 * out["emissions_kg"] / out["emissions_kg"].sum()
    return out


# ---- selection helpers ------------------------------------------------------
def normalise_show(show: Show) -> list[str] | None:
    """
    Turn a stage selection into a list of stage strings.

    Parameters
    ----------
    show : str, sequence of str or None
        ``None`` or ``"all"`` for everything, a single stage (``"3"``,
        ``"2.4"``) or a list of them.

    Returns
    -------
    list of str or None
        ``None`` means everything.
    """
    if show is None:
        return None
    items = [show] if isinstance(show, str) else [str(s) for s in show]
    return None if "all" in items else items


def stage_mask(ts: pd.DataFrame, show: Show = None) -> pd.Series:
    """
    Boolean mask of the samples in a stage selection.

    Parameters
    ----------
    ts : pandas.DataFrame
        Output of :func:`load_timeseries`.
    show : str, sequence of str or None
        Major stages (``"3"``) and/or exact stage IDs (``"2.4"``); see
        :func:`normalise_show`.

    Returns
    -------
    pandas.Series of bool
        Aligned with ``ts``.
    """
    items = normalise_show(show)
    if items is None:
        return pd.Series(True, index=ts.index)
    return ts["major"].isin(items) | ts["stage_id"].isin(items)


def describe_selection(show: Show) -> str:
    """
    Readable name for a stage selection, for plot titles.

    Parameters
    ----------
    show : str, sequence of str or None
        See :func:`normalise_show`.

    Returns
    -------
    str
        E.g. ``"3 final training, 2.4"`` or ``"all stages"``.
    """
    items = normalise_show(show)
    if items is None:
        return "all stages"
    return ", ".join(STAGE_COLOURS[s][1] if s in STAGE_COLOURS else s for s in items)


# ---- plot helpers -----------------------------------------------------------
def _style(ax: Axes, grid_axis: Literal["x", "y", "both"] = "y") -> None:
    """
    Apply the shared axis style: no top/right spines, light recessive grid.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to style.
    grid_axis : {"x", "y", "both"}, default "y"
        Which gridlines to draw.
    """
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _stage_legend(fig: Figure, majors: Sequence[str] | pd.Series, power: PowerView = "machine") -> None:
    """
    Add the stage-colour legend across the top of a figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to add the legend to.
    majors : sequence of str
        Major stages present in the run; only these get legend entries.
    power : {"machine", "process", "both"}, default "machine"
        With ``"both"``, adds entries explaining solid (process) and light
        (rest of machine) fills (used for ``process_style="layers"``).
    """
    present = [m for m in STAGE_COLOURS if m in set(majors)]
    handles: list = [plt.Rectangle((0, 0), 1, 1, color=STAGE_COLOURS[m][0]) for m in present]
    labels = [STAGE_COLOURS[m][1] for m in present]
    if power == "both":
        handles += [plt.Rectangle((0, 0), 1, 1, color=INK_MUTED), plt.Rectangle((0, 0), 1, 1, color=INK_MUTED, alpha=REST_ALPHA)]
        labels += ["this process", "rest of machine"]
    fig.legend(handles, labels, ncol=len(labels), frameon=False, loc="upper center")


def _smooth(y: pd.Series, smooth: int) -> pd.Series:
    """
    Centred rolling mean.

    Parameters
    ----------
    y : pandas.Series
        Values to smooth.
    smooth : int
        Window in samples; 1 or less returns ``y`` unchanged.

    Returns
    -------
    pandas.Series
        Smoothed values.
    """
    return y.rolling(smooth, min_periods=1, center=True).mean() if smooth > 1 else y


def _require_process(ts: pd.DataFrame, power: PowerView) -> None:
    """
    Raise a clear error when a process view is asked of a run without process data.

    Parameters
    ----------
    ts : pandas.DataFrame
        One run.
    power : {"machine", "process", "both"}
        Requested view.

    Raises
    ------
    ValueError
        If ``power`` needs process data that the run does not have.
    """
    if power != "machine" and not has_process_data(ts):
        raise ValueError(
            "This run has no process data (recorded before process attribution was added); use power='machine'."
        )


# ---- plots ------------------------------------------------------------------
def plot_overview(
    ts: pd.DataFrame,
    show: Show = None,
    power: PowerView = "machine",
    panels: Sequence[Panel] = ("power", "cumulative", "shares"),
    process_style: ProcessStyle = "hatch",
    context: Literal["hidden", "ghost"] = "hidden",
    x_range: Literal["full", "selection"] = "full",
    x_origin: Literal["run", "selection"] = "run",
    time_unit: TimeUnit = "min",
    smooth: int = 1,
    y_max: float | None = None,
    y_scale: Literal["run", "shown"] = "run",
    hatch: str = "///",
    process_alpha: float = 0.45,
    linewidth: float = LINEWIDTH,
    title: str | None = None,
    legend: bool = True,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """
    Power over time, cumulative emissions and emissions per major stage.

    By default all value axes use the full run's *machine* ranges whatever is
    selected or shown, so slides with different selections, panels or power
    views share the same scales; ``y_scale="shown"`` fits them to what is
    drawn instead.

    The process's power is always drawn the same way (with the default
    ``process_style="hatch"``): lighter stage colour, hatched and outlined,
    whether shown alone or stacked on top of the rest of the machine (solid).

    Parameters
    ----------
    ts : pandas.DataFrame
        One run from :func:`load_timeseries`.
    show : str, sequence of str or None, default None
        Stages to include: ``None``/``"all"`` for everything, a major stage
        (``"3"`` = final training), a stage ID (``"2.4"`` = one HPO trial) or a
        list of them, e.g. ``["3", "2"]``.
    power : {"machine", "process", "both"}, default "machine"
        ``"machine"``: whole-computer power. ``"process"``: only the share
        attributable to this Python process. ``"both"``: the rest of the
        machine with the process stacked on top, drawn according to
        ``process_style``.
    panels : sequence of {"power", "cumulative", "shares"}
        Which panels to draw, top to bottom: power over time, cumulative
        emissions over time, and emissions per major stage.
    process_style : {"hatch", "layers"}, default "hatch"
        How the process is drawn. ``"hatch"``: lighter stage colour, hatched
        and outlined, in both ``power="process"`` and ``power="both"``; in
        ``"both"`` it sits on top of the rest of the machine (solid), as if
        adding to what was already running, so the silhouette matches
        ``power="machine"``. ``"layers"``: process in solid stage colour (at
        the bottom in ``"both"``) with the rest of the machine as a lighter
        layer above it.
    context : {"hidden", "ghost"}, default "hidden"
        Unselected samples are left blank (``"hidden"``) or drawn in grey.
    x_range : {"full", "selection"}, default "full"
        Time axis covers the whole run or zooms to the selected samples.
    x_origin : {"run", "selection"}, default "run"
        Time measured from the tracker start or from the selection's start.
    time_unit : {"s", "min", "h"}, default "min"
        Unit of the time axis.
    smooth : int, default 1
        Rolling-mean window (samples) for the power panel.
    y_max : float, optional
        Fixed upper limit of the power axis (W); overrides ``y_scale`` there.
    y_scale : {"run", "shown"}, default "run"
        ``"run"``: value axes (power, cumulative emissions, per-stage bars)
        span the full run's machine values, identical on every slide.
        ``"shown"``: each axis fits what this plot draws, e.g. just the
        process's power during the selected stage.
    hatch : str, default "///"
        Matplotlib hatch pattern for ``process_style="hatch"``; repeat the
        characters for denser lines (e.g. ``"////"``).
    process_alpha : float, default 0.45
        Opacity of the stage colour under the hatching of the process's share
        (``process_style="hatch"``); lower is lighter.
    linewidth : float, default 0.5
        Width (points) of every line drawn: hatch lines, the outline around
        the process's share and the cumulative-emissions lines.
    title : str, optional
        Title for the top panel; no title by default.
    legend : bool, default True
        Draw the stage legend (and the "this process" key). Turn off when a
        single stage is shown and the colours need no explanation.
    figsize : tuple of float, optional
        Defaults to 11 in wide with height scaled to the panels drawn.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Raises
    ------
    ValueError
        If ``panels`` is empty or contains an unknown name, or a process view
        is requested for a run without process data.

    Examples
    --------
    Final training only, time axis zoomed to it, just the power panel:

    >>> plot_overview(ts, show="3", x_range="selection", panels=["power"])  # doctest: +SKIP

    The same slide again with the process's share hatched:

    >>> plot_overview(ts, show="3", x_range="selection", panels=["power"], power="both")  # doctest: +SKIP

    Process only, with the axis fitted to it:

    >>> plot_overview(ts, show="3", x_range="selection", power="process", y_scale="shown")  # doctest: +SKIP
    """
    panels = list(panels)
    if not panels or set(panels) - set(PANEL_HEIGHTS):
        raise ValueError(f"panels must be a non-empty subset of {list(PANEL_HEIGHTS)}, got {panels}")
    _require_process(ts, power)
    styled = power != "machine" and process_style == "hatch"  # process drawn hatched + outlined
    hatched = styled and power == "both"
    layered = power == "both" and process_style == "layers"
    fit = y_scale == "shown"

    k = TIME_UNITS[time_unit]
    mask = stage_mask(ts, show)
    sel_start, sel_end = ts.loc[mask, "t_start_s"].min(), ts.loc[mask, "elapsed_s"].max()
    origin = sel_start if x_origin == "selection" else 0.0
    t0, t1 = (ts["t_start_s"] - origin) / k, (ts["elapsed_s"] - origin) / k
    if x_range == "selection":
        x_lim = ((sel_start - origin) / k, (sel_end - origin) / k)
    else:
        x_lim = (-origin / k, (ts["elapsed_s"].max() - origin) / k)
    x_label = f"Time ({time_unit})"
    majors = [m for m in STAGE_COLOURS if m in set(ts["major"])]

    heights = [PANEL_HEIGHTS[p] for p in panels]
    if figsize is None:
        figsize = (11, 0.8 + 1.55 * sum(heights))
    fig, axes = plt.subplots(len(panels), 1, figsize=figsize, gridspec_kw={"height_ratios": heights}, squeeze=False)
    ax_of = dict(zip(panels, axes[:, 0]))
    time_panels = [p for p in panels if p in ("power", "cumulative")]

    def colour(m: str) -> str:
        """
        Fill colour of a major stage.

        Parameters
        ----------
        m : str
            Major stage, e.g. ``"3"``.

        Returns
        -------
        str
            Hex colour; grey for unknown stages.
        """
        return STAGE_COLOURS.get(m, ("#888888",))[0]

    def process_fill(
        ax: Axes,
        x: Sequence[float],
        y0: float | Sequence[float],
        y1: float | Sequence[float],
        fill: str | None,
        where: Sequence[bool] | None = None,
    ) -> None:
        """
        Draw the process's share: light stage colour with hatching.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to draw on.
        x : sequence of float
            x coordinates.
        y0, y1 : float or sequence of float
            Lower and upper bounds of the area.
        fill : str or None
            Base colour, drawn at ``process_alpha`` opacity under the hatching;
            None for hatching only (whatever is underneath shows through).
        where : sequence of bool, optional
            Only draw where True.
        """
        facecolor = "none" if fill is None else to_rgba(fill, process_alpha)
        ax.fill_between(x, y0, y1, where=where, facecolor=facecolor, edgecolor=HATCH_COLOUR, hatch=hatch, lw=0)

    def outline(ax: Axes, values: pd.Series, keep: pd.Series) -> None:
        """
        Draw a stepped line along per-sample values, over selected samples only.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to draw on.
        values : pandas.Series
            One value per sample (aligned with ``ts``).
        keep : pandas.Series of bool
            Samples to draw; the line is broken at the others and at
            unselected samples.
        """
        xs: list[float] = []
        ys: list[float] = []
        for a, b, v, shown, k in zip(t0, t1, values, mask, keep):
            if shown and k:
                xs += [a, b]
                ys += [v, v]
            elif xs and xs[-1] == xs[-1]:  # break the line (NaN) once per gap
                xs.append(float("nan"))
                ys.append(float("nan"))
        ax.plot(xs, ys, color=HATCH_COLOUR, lw=linewidth, solid_joinstyle="miter")

    with plt.rc_context({"hatch.linewidth": linewidth}):
        if "power" in ax_of:
            ax = ax_of["power"]
            machine = _smooth(ts["power_w"], smooth)
            process = _smooth(ts["proc_power_w"], smooth) if power != "machine" else None
            for i, (a, b, m, shown) in enumerate(zip(t0, t1, ts["major"], mask)):
                if not shown:
                    if context == "ghost":
                        ax.fill_between([a, b], 0, machine.iloc[i], color=GHOST, lw=0)
                    continue
                if power == "machine":
                    ax.fill_between([a, b], 0, machine.iloc[i], color=colour(m), lw=0)
                elif power == "process":
                    if styled:
                        process_fill(ax, [a, b], 0, process.iloc[i], colour(m))
                    else:
                        ax.fill_between([a, b], 0, process.iloc[i], color=colour(m), lw=0)
                elif hatched:  # rest of machine at the bottom, process stacked on top
                    rest = max(machine.iloc[i] - process.iloc[i], 0.0)
                    ax.fill_between([a, b], 0, rest, color=colour(m), lw=0)
                    process_fill(ax, [a, b], rest, machine.iloc[i], colour(m))
                else:
                    ax.fill_between([a, b], process.iloc[i], max(machine.iloc[i], process.iloc[i]),
                                    color=colour(m), alpha=REST_ALPHA, lw=0)
                    ax.fill_between([a, b], 0, process.iloc[i], color=colour(m), lw=0)
            if hatched:  # outline the process's share: its lower edge and the top edge
                has_power = machine > 0  # no outline along the baseline for empty samples
                outline(ax, (machine - process).clip(lower=0), has_power)
                outline(ax, machine, has_power)
            elif styled:  # process alone: outline its top edge
                outline(ax, process, process > 0)
            if fit:
                drawn = process if power == "process" else machine
                top = drawn[mask].max()
                if context == "ghost":
                    top = max(top, machine[~mask].max() if (~mask).any() else 0)
            else:
                top = machine.max()
            ax.set_ylabel("Power (W)")
            ax.set_ylim(0, y_max if y_max else top * 1.05)
            ax.set_xlim(*x_lim)

        if "cumulative" in ax_of:
            ax = ax_of["cumulative"]
            if context == "ghost":
                ax.plot(t1, ts["emissions_kg"].cumsum() * 1000, color=GHOST, lw=linewidth, ls="--")
            machine_cum = ts["emissions_kg"].where(mask, 0).cumsum() * 1000
            process_cum = ts["proc_emissions_kg"].where(mask, 0).cumsum() * 1000 if power != "machine" else None
            top_cum = process_cum if power == "process" else machine_cum

            # stage colour under the line only, one fill per contiguous stage segment
            # (starting at the previous sample so neighbouring segments meet)
            segments = (ts["major"] != ts["major"].shift()).cumsum()
            for _, seg in ts[mask].groupby(segments[mask]):
                idx = list(seg.index)
                first = idx[0]
                x0 = t0.iloc[first]
                y0 = top_cum.iloc[first - 1] if first > 0 else 0.0
                ax.fill_between([x0, *t1.iloc[idx]], 0, [y0, *top_cum.iloc[idx]],
                                color=colour(seg["major"].iloc[0]), alpha=CUMULATIVE_ALPHA, lw=0)

            if power == "machine":
                ax.plot(t1, machine_cum, color=INK_MUTED, lw=linewidth)
            elif hatched:  # process's emissions stacked on top of the rest of the machine
                rest_cum = machine_cum - process_cum
                process_fill(ax, t1, rest_cum, machine_cum, None, where=mask)
                ax.plot(t1, rest_cum, color=HATCH_COLOUR, lw=linewidth)
                ax.plot(t1, machine_cum, color=INK_MUTED, lw=linewidth)
            elif styled:  # process alone, drawn like its share in the stacked view
                process_fill(ax, t1, 0, process_cum, None, where=mask)
                ax.plot(t1, process_cum, color=HATCH_COLOUR, lw=linewidth)
            else:
                if layered:
                    ax.plot(t1, machine_cum, color=INK_MUTED, lw=linewidth, ls="--", label="machine")
                ax.plot(t1, process_cum, color=INK, lw=linewidth, label="process-specific")
                if layered and legend:
                    ax.legend(frameon=False, loc="upper left")
            ax.set_ylabel("Cumulative emissions (g CO₂e)")
            ax.set_xlim(*x_lim)
            if fit:
                cum_top = (process_cum if power == "process" else machine_cum).max()
                if context == "ghost":
                    cum_top = max(cum_top, ts["emissions_kg"].sum() * 1000)
            else:
                cum_top = ts["emissions_kg"].sum() * 1000
            ax.set_ylim(0, cum_top * 1.05)

        if "shares" in ax_of:
            ax = ax_of["shares"]
            full_machine = ts.groupby("major")["emissions_kg"].sum() * 1000
            shown_machine = ts[mask].groupby("major")["emissions_kg"].sum() * 1000
            shown_process = ts[mask].groupby("major")["proc_emissions_kg"].sum() * 1000 if power != "machine" else None
            items = normalise_show(show)
            rows = []  # (label, base value, base colour, extra value, extra colour, text)
            for m in majors:
                name = STAGE_COLOURS[m][1]
                if items is not None and m not in items and m in shown_machine.index:
                    ids = sorted(set(ts.loc[mask & (ts["major"] == m), "stage_id"]))
                    name += f" ({', '.join(ids)})"  # only some steps of this stage selected
                if m in shown_machine.index:
                    mach = shown_machine[m]
                    proc = shown_process[m] if shown_process is not None else 0.0
                    if power == "machine":
                        rows.append((name, mach, colour(m), 0, colour(m), f"{mach:.3f} g"))
                    elif power == "process" and styled:  # same hatched box as in the stacked view
                        rows.append((name, 0, colour(m), proc, colour(m), f"{proc:.3f} g"))
                    elif power == "process":
                        rows.append((name, proc, colour(m), 0, colour(m), f"{proc:.3f} g"))
                    elif hatched:  # rest of machine first, process stacked on the end
                        rows.append((name, max(mach - proc, 0), colour(m), min(proc, mach), colour(m),
                                     f"{mach:.3f} g (process: {proc:.3f} g)"))
                    else:
                        rows.append((name, proc, colour(m), max(mach - proc, 0), colour(m), f"{proc:.3f} of {mach:.3f} g"))
                elif context == "ghost":
                    rows.append((name, full_machine[m], GHOST, 0, GHOST, ""))
                else:
                    rows.append((name, 0, "none", 0, "none", ""))
            labels, base, base_c, extra, extra_c, texts = map(list, zip(*rows))
            ax.barh(labels, base, color=base_c, height=0.6)
            if styled:
                clear = (0.0, 0.0, 0.0, 0.0)
                bars = ax.barh(
                    labels, extra, left=base, height=0.6, hatch=hatch, lw=linewidth,
                    facecolor=[to_rgba(c, process_alpha) if v > 0 else clear for c, v in zip(extra_c, extra)],
                    edgecolor=[HATCH_COLOUR if v > 0 else clear for v in extra],
                )
            else:
                bars = ax.barh(labels, extra, left=base, color=extra_c, alpha=REST_ALPHA, height=0.6)
            ax.bar_label(bars, labels=texts, padding=4, color=INK)
            ax.invert_yaxis()
            bar_top = max(b + e for b, e in zip(base, extra)) if fit else full_machine.max()
            ax.set_xlim(0, (bar_top or full_machine.max()) * 1.3)
            ax.set_xlabel("Emissions per stage (g CO₂e)")

        if styled and legend:  # own legend, so the stage legend is identical to the machine view
            key = Patch(facecolor=to_rgba(INK_MUTED, process_alpha * 0.4), edgecolor=HATCH_COLOUR, hatch=hatch,
                        lw=linewidth, label="Process-specific")
            # in the figure's top band (space already reserved for the stage legend),
            # so the axes keep exactly the same size as on the machine-only slide
            fig.legend(handles=[key], frameon=False, loc="upper right")

    if time_panels:
        ax_of[time_panels[-1]].set_xlabel(x_label)
    if title:
        ax_of[panels[0]].set_title(title, loc="left")
    for name, ax in ax_of.items():
        _style(ax, "x" if name == "shares" else "y")
    if legend:
        _stage_legend(fig, ts["major"], "both" if layered else "machine")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / figsize[1]) if legend else None)
    # tight_layout sizes the left margin to the widest tick label ("8" vs "14"),
    # which would nudge the plot sideways between slides; use a fixed minimum
    fig.subplots_adjust(left=max(fig.subplotpars.left, MIN_LEFT_MARGIN_IN / figsize[0]))
    return fig


def reveal_steps(order: Sequence[str], cumulative: bool = True) -> list[list[str]]:
    """
    Expand a reveal order into per-slide stage selections.

    Parameters
    ----------
    order : sequence of str
        Stages in reveal order, e.g. ``["3", "2", "1"]``; ``"all"`` allowed.
    cumulative : bool, default True
        Each step adds to the previous ones; otherwise one stage per step.

    Returns
    -------
    list of list of str
        E.g. ``[["3"], ["3", "2"], ["3", "2", "1"]]``.
    """
    order = [str(o) for o in order]
    return [order[: i + 1] for i in range(len(order))] if cumulative else [[o] for o in order]


def save_reveal_sequence(
    ts: pd.DataFrame,
    order: Sequence[str],
    out_dir: str | Path,
    prefix: str = "reveal",
    cumulative: bool = True,
    panels_per_step: Sequence[Sequence[Panel]] | None = None,
    power_per_step: Sequence[PowerView] | None = None,
    dpi: int = 200,
    **plot_kwargs,
) -> list[Path]:
    """
    Save one :func:`plot_overview` PNG per reveal step.

    Parameters
    ----------
    ts : pandas.DataFrame
        One run from :func:`load_timeseries`.
    order : sequence of str
        Reveal order, e.g. ``["3", "all"]`` for a final-training slide then a
        whole-run slide. A stage may repeat, e.g. ``["3", "3"]`` with
        ``power_per_step=["machine", "both"]`` for the same slide without and
        then with the process highlighted.
    out_dir : str or Path
        Folder for the PNGs (created if needed).
    prefix : str, default "reveal"
        File-name prefix.
    cumulative : bool, default True
        See :func:`reveal_steps`.
    panels_per_step : sequence of sequence of str, optional
        Panels for each step, e.g. ``[["power"], ["power", "cumulative", "shares"]]``.
        Defaults to ``plot_kwargs["panels"]`` (or all panels) for every step.
    power_per_step : sequence of {"machine", "process", "both"}, optional
        Power view for each step. Defaults to ``plot_kwargs["power"]`` (or
        ``"machine"``) for every step.
    dpi : int, default 200
        Resolution of the PNGs.
    **plot_kwargs
        Passed to :func:`plot_overview` (e.g. ``x_range="selection"``,
        ``process_style="hatch"``).

    Returns
    -------
    list of Path
        The files written, in order.

    Raises
    ------
    ValueError
        If ``panels_per_step`` or ``power_per_step`` does not have one entry
        per step.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = reveal_steps(order, cumulative)
    for name, per_step in (("panels_per_step", panels_per_step), ("power_per_step", power_per_step)):
        if per_step is not None and len(per_step) != len(steps):
            raise ValueError(f"{name} has {len(per_step)} entries but there are {len(steps)} steps")
    default_panels = plot_kwargs.pop("panels", ("power", "cumulative", "shares"))
    default_power = plot_kwargs.pop("power", "machine")
    paths = []
    for i, show in enumerate(steps, start=1):
        panels = panels_per_step[i - 1] if panels_per_step is not None else default_panels
        power = power_per_step[i - 1] if power_per_step is not None else default_power
        fig = plot_overview(ts, show=show, panels=panels, power=power, **plot_kwargs)
        items = normalise_show(show)
        stem = "all" if items is None else "-".join(dict.fromkeys(items))
        path = out_dir / f"{prefix}_{i:02d}_{stem}_{power}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        paths.append(path)
    return paths


def save_slides(
    ts: pd.DataFrame,
    slides: Sequence[Mapping[str, Any]],
    out_dir: str | Path,
    prefix: str = "slide",
    dpi: int = 200,
    **common: Any,
) -> list[Path]:
    """
    Save a sequence of :func:`plot_overview` PNGs, one per slide.

    Parameters
    ----------
    ts : pandas.DataFrame
        One run from :func:`load_timeseries`.
    slides : sequence of mapping
        One dict of :func:`plot_overview` keyword arguments per slide; these
        override ``common`` for that slide.
    out_dir : str or Path
        Folder for the PNGs (created if needed).
    prefix : str, default "slide"
        File-name prefix; files are ``<prefix>_01.png``, ``<prefix>_02.png``, ...
    dpi : int, default 200
        Resolution of the PNGs.
    **common
        Keyword arguments for :func:`plot_overview` shared by every slide.

    Returns
    -------
    list of Path
        The files written, in order.

    Examples
    --------
    Process power during final training, then the whole machine during final
    training, then the whole run:

    >>> save_slides(ts, [  # doctest: +SKIP
    ...     dict(show="3", power="process", x_range="selection", y_scale="shown"),
    ...     dict(show="3", power="both", x_range="selection"),
    ...     dict(show="all", power="both"),
    ... ], "outputs/slides", legend=False)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, slide in enumerate(slides, start=1):
        fig = plot_overview(ts, **{**common, **slide})
        path = out_dir / f"{prefix}_{i:02d}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_power_components(
    ts: pd.DataFrame,
    time_unit: TimeUnit = "min",
    smooth: int = 1,
    show_process: bool = True,
    figsize: tuple[float, float] = (11, 3.5),
) -> Figure:
    """
    Stacked machine power by hardware component, with process power overlaid.

    Parameters
    ----------
    ts : pandas.DataFrame
        One run from :func:`load_timeseries`.
    time_unit : {"s", "min", "h"}, default "min"
        Unit of the time axis.
    smooth : int, default 1
        Rolling-mean window in samples.
    show_process : bool, default True
        Overlay this process's attributed power as a line, when recorded.
    figsize : tuple of float, default (11, 3.5)
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    k = TIME_UNITS[time_unit]
    t = ts["elapsed_s"] / k
    parts = {c: _smooth(ts[f"{c}_w"], smooth) for c in ("cpu", "gpu", "ram")}
    fig, ax = plt.subplots(figsize=figsize)
    ax.stackplot(t, *parts.values(), labels=["CPU", "GPU", "RAM (fixed estimate)"],
                 colors=list(COMPONENT_COLOURS.values()), step="pre", lw=0)
    if show_process and has_process_data(ts):
        ax.step(t, _smooth(ts["proc_power_w"], smooth), where="pre", color=INK, lw=1.5, label="process")
    ax.set_xlim(0, t.max())
    ax.set_ylim(0, None)
    ax.set_ylabel("Power (W)")
    ax.set_xlabel(f"Elapsed time ({time_unit})")
    ax.set_title("Machine power by component", loc="left")
    ax.legend(frameon=False, ncol=4, loc="upper left")
    _style(ax)
    fig.tight_layout()
    return fig


def plot_hpo_cost(
    ts: pd.DataFrame,
    hpo: pd.DataFrame,
    power: Literal["machine", "process"] = "machine",
    figsize: tuple[float, float] = (7, 4.5),
) -> tuple[Figure, pd.DataFrame]:
    """
    Emissions of each HPO trial against its validation AUC.

    Parameters
    ----------
    ts : pandas.DataFrame
        One run from :func:`load_timeseries`.
    hpo : pandas.DataFrame
        That run's ``hpo_results_<run>.csv`` (``stage_id``, hyperparameters,
        ``val_auc``).
    power : {"machine", "process"}, default "machine"
        Which emissions to plot.
    figsize : tuple of float, default (7, 4.5)
        Figure size in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The scatter plot.
    trials : pandas.DataFrame
        One row per trial: hyperparameters, validation AUC, duration, energy
        and emissions.
    """
    _require_process(ts, power)
    trials = stage_summary(ts).merge(hpo, on="stage_id")
    col = "proc_emissions_kg" if power == "process" else "emissions_kg"
    param_cols = [c for c in hpo.columns if c not in ("stage_id", "val_auc")]
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(trials[col] * 1000, trials["val_auc"], s=60,
               color=STAGE_COLOURS["2"][0], edgecolor="white", linewidth=2, zorder=3)
    for _, r in trials.iterrows():
        ax.annotate(r["stage_id"], (r[col] * 1000, r["val_auc"]), xytext=(6, 4),
                    textcoords="offset points", color=INK_MUTED, fontsize=9)
    ax.set_xlabel(f"Trial emissions, {power} (g CO₂e)")
    ax.set_ylabel("Validation AUC")
    ax.set_title("HPO trials: cost vs performance", loc="left")
    _style(ax, "both")
    fig.tight_layout()
    keep = ["stage_id", *param_cols, "val_auc", "duration_s", "energy_kwh", "emissions_kg"]
    if power == "process":
        keep += ["proc_energy_kwh", "proc_emissions_kg"]
    return fig, trials[keep]


def plot_energy_coverage(ts: pd.DataFrame, time_unit: TimeUnit = "min", figsize: tuple[float, float] = (11, 2.8)) -> Figure:
    """
    Fraction of each interval that codecarbon counted energy for.

    Parameters
    ----------
    ts : pandas.DataFrame
        One run from :func:`load_timeseries`.
    time_unit : {"s", "min", "h"}, default "min"
        Unit of the time axis.
    figsize : tuple of float, default (11, 2.8)
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        Line plot of ``energy_coverage`` (%) with a 100% reference line.
    """
    k = TIME_UNITS[time_unit]
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(ts["elapsed_s"] / k, ts["energy_coverage"] * 100, color=INK_MUTED, lw=1.5)
    ax.axhline(100, color=STAGE_COLOURS["2"][0], lw=1, ls="--")
    ax.set_ylim(0, 110)
    ax.set_xlim(0, ts["elapsed_s"].max() / k)
    ax.set_ylabel("Time counted (%)")
    ax.set_xlabel(f"Elapsed time ({time_unit})")
    ax.set_title("Energy coverage: share of each interval codecarbon counted energy for", loc="left")
    _style(ax)
    fig.tight_layout()
    return fig
