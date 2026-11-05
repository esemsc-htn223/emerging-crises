"""
Loading and plotting helpers for the stage-tagged emissions timeseries.

Read-only: nothing here imports codecarbon or starts a tracker, so it is safe
to use for exploring results without collecting new data.

Power columns
-------------
codecarbon's ``cpu_power_w`` / ``gpu_power_w`` / ``ram_power_w`` are *run-to-date
averages* (sum of readings / number of readings), not the power in each
interval, so they ramp smoothly and are unsuitable for a timeseries.

``load_timeseries`` adds true per-interval power from the interval energies:
``cpu_w``, ``gpu_w``, ``ram_w`` and ``power_w`` (total) = energy / duration.
Very short samples (< ``MIN_SAMPLE_S``) at stage boundaries can carry energy
belonging to the neighbouring interval (codecarbon's scheduler thread and
``flush()`` race), which would show as huge spikes, so they are merged into the
preceding sample for the power calculation. Energy/emission totals are unchanged.

``energy_coverage`` = RAM interval power / RAM nominal power. RAM is a constant
estimate, so this is the fraction of wall time codecarbon actually counted
energy for. It is ~1 in cpu_load mode, but ~0.5 with Apple PowerMetrics because
the time spent running ``powermetrics`` itself (~2.5 s per sample) is dropped.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# one colour per major stage, fixed order (never cycled)
STAGE_COLOURS = {
    "0": ("#2a78d6", "0 setup"),
    "1": ("#eb6834", "1 data"),
    "2": ("#1baf7a", "2 HPO"),
    "3": ("#eda100", "3 final training"),
    "4": ("#e87ba4", "4 application"),
}
COMPONENT_COLOURS = {"cpu": "#2a78d6", "gpu": "#eb6834", "ram": "#1baf7a"}
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e5e4df"
TIME_UNITS = {"s": 1.0, "min": 60.0, "h": 3600.0}
MIN_SAMPLE_S = 0.5


# ---- data -------------------------------------------------------------------
def load_timeseries(path: str | Path) -> pd.DataFrame:
    """Read emissions_timeseries.csv (all runs) and add derived columns."""
    df = pd.read_csv(path, dtype={"stage_id": str, "run_id": str})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["major"] = df["stage_id"].str.split(".").str[0]
    df["t_start_s"] = df["elapsed_s"] - df["duration_s"]

    # per-interval power; short samples join the preceding sample of the same run
    block = (df["duration_s"] >= MIN_SAMPLE_S).groupby(df["run_id"]).cumsum()
    grp = df.groupby([df["run_id"], block])
    dur = grp["duration_s"].transform("sum")
    for part, col in (("cpu", "cpu_energy_kwh"), ("gpu", "gpu_energy_kwh"), ("ram", "ram_energy_kwh"), ("power", "energy_kwh")):
        name = "power_w" if part == "power" else f"{part}_w"
        df[name] = (grp[col].transform("sum") * 3.6e6 / dur).where(dur > 0, 0.0)

    ram_nominal = df.groupby("run_id")["ram_power_w"].transform("median")
    df["energy_coverage"] = df["ram_w"] / ram_nominal
    return df


def list_runs(df: pd.DataFrame) -> pd.DataFrame:
    """One row per run_id, longest first."""
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
    return runs.sort_values("duration_min", ascending=False).reset_index(drop=True)


def select_run(df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    return df[df["run_id"] == run_id].reset_index(drop=True)


def stage_summary(ts: pd.DataFrame) -> pd.DataFrame:
    """Energy, emissions and duration per stage ID."""
    out = (
        ts.groupby(["stage_id", "stage"], sort=False)
        .agg(
            major_stage=("major", "first"),
            duration_s=("duration_s", "sum"),
            n_samples=("duration_s", "size"),
            energy_kwh=("energy_kwh", "sum"),
            emissions_kg=("emissions_kg", "sum"),
            cpu_energy_kwh=("cpu_energy_kwh", "sum"),
            energy_coverage=("energy_coverage", "median"),
        )
        .reset_index()
    )
    out["mean_power_w"] = out["energy_kwh"] * 3.6e6 / out["duration_s"]
    out["mean_cpu_power_w"] = out["cpu_energy_kwh"] * 3.6e6 / out["duration_s"]
    out = out.drop(columns="cpu_energy_kwh")
    out["share_of_emissions_pct"] = 100 * out["emissions_kg"] / out["emissions_kg"].sum()
    return out


def major_summary(ts: pd.DataFrame) -> pd.DataFrame:
    out = (
        ts.groupby("major")
        .agg(duration_s=("duration_s", "sum"), energy_kwh=("energy_kwh", "sum"), emissions_kg=("emissions_kg", "sum"))
        .reset_index()
    )
    out.insert(1, "stage", out["major"].map(lambda m: STAGE_COLOURS.get(m, ("", m))[1]))
    out["share_of_emissions_pct"] = 100 * out["emissions_kg"] / out["emissions_kg"].sum()
    return out


# ---- plots ------------------------------------------------------------------
def _style(ax, grid_axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _stage_legend(fig, majors):
    present = [m for m in STAGE_COLOURS if m in set(majors)]
    handles = [plt.Rectangle((0, 0), 1, 1, color=STAGE_COLOURS[m][0]) for m in present]
    fig.legend(handles, [STAGE_COLOURS[m][1] for m in present], ncol=len(present), frameon=False, loc="upper center")


def _power_series(ts, smooth):
    y = ts["power_w"]
    if smooth > 1:
        y = y.rolling(smooth, min_periods=1, center=True).mean()
    return y


GHOST = "#d9d8d3"  # hidden-stage context colour


def stage_mask(ts: pd.DataFrame, show=None) -> pd.Series:
    """
    Boolean mask of samples to show. ``show`` is None (everything) or a list of
    major stages ("3") and/or exact stage IDs ("2.4"), e.g. ["3", "2.1", "2.2"].
    """
    if show is None:
        return pd.Series(True, index=ts.index)
    show = [str(s) for s in show]
    return ts["major"].isin(show) | ts["stage_id"].isin(show)


def describe_selection(show) -> str:
    if show is None:
        return "all stages"
    names = [STAGE_COLOURS[s][1] if s in STAGE_COLOURS else s for s in map(str, show)]
    return ", ".join(names)


def plot_overview(
    ts: pd.DataFrame,
    show=None,
    context: str = "hidden",
    x_range: str = "full",
    x_origin: str = "run",
    time_unit: str = "min",
    smooth: int = 1,
    y_max: float | None = None,
    title: str | None = None,
    figsize=(11, 9),
):
    """
    Power over time, cumulative emissions and emissions per major stage.

    show     None for everything, or a list of major stages / stage IDs to
             include, e.g. ["3"] (final training only) or ["3", "2"].
    context  "hidden": unselected samples are not drawn. "ghost": drawn in grey.
    x_range  "full": the whole run's time axis. "selection": zoom the time axis
             to the selected samples (e.g. just final training).
    x_origin "run": time since the tracker started. "selection": time since the
             start of the selection (axis starts at 0).
    The y-axes (power, cumulative emissions, per-stage bars) always use the full
    run's ranges, so slides with different selections share the same scales.
    """
    k = TIME_UNITS[time_unit]
    mask = stage_mask(ts, show)
    sel_start, sel_end = ts.loc[mask, "t_start_s"].min(), ts.loc[mask, "elapsed_s"].max()
    origin = sel_start if x_origin == "selection" else 0.0
    t0, t1 = (ts["t_start_s"] - origin) / k, (ts["elapsed_s"] - origin) / k
    if x_range == "selection":
        x_lim = ((sel_start - origin) / k, (sel_end - origin) / k)
    else:
        x_lim = (-origin / k, (ts["elapsed_s"].max() - origin) / k)
    x_label = f"Time since start of selection ({time_unit})" if x_origin == "selection" else f"Elapsed time ({time_unit})"
    y = _power_series(ts, smooth)
    total_g = ts["emissions_kg"].sum() * 1000
    by_major_full = ts.groupby("major")["emissions_kg"].sum() * 1000
    majors = [m for m in STAGE_COLOURS if m in set(ts["major"])]

    fig, (ax_p, ax_c, ax_b) = plt.subplots(3, 1, figsize=figsize, gridspec_kw={"height_ratios": [2, 2, 1.4]})

    # power: one filled block per sample, coloured by stage
    for a, b, v, m, shown in zip(t0, t1, y, ts["major"], mask):
        if shown:
            ax_p.fill_between([a, b], 0, v, color=STAGE_COLOURS.get(m, ("#888888",))[0], lw=0)
        elif context == "ghost":
            ax_p.fill_between([a, b], 0, v, color=GHOST, lw=0)
    ax_p.set_ylabel("Power (W)")
    ax_p.set_ylim(0, y_max if y_max else y.max() * 1.05)
    ax_p.set_xlim(*x_lim)
    ax_p.set_title(title or f"Power draw over the workflow: {describe_selection(show)}", loc="left")

    # cumulative emissions of the selected samples only
    if context == "ghost":
        ax_c.plot(t1, ts["emissions_kg"].cumsum() * 1000, color=GHOST, lw=2, ls="--")
    ax_c.plot(t1, ts["emissions_kg"].where(mask, 0).cumsum() * 1000, color=INK_MUTED, lw=2)
    runs = (ts["major"] != ts["major"].shift()).cumsum()
    for _, seg in ts[mask].groupby(runs[mask]):
        colour = STAGE_COLOURS.get(seg["major"].iloc[0], ("#888888",))[0]
        ax_c.axvspan((seg["t_start_s"].min() - origin) / k, (seg["elapsed_s"].max() - origin) / k, color=colour, alpha=0.18, lw=0)
    ax_c.set_ylabel("Cumulative emissions (g CO₂e)")
    ax_c.set_xlabel(x_label)
    ax_c.set_xlim(*x_lim)
    ax_c.set_ylim(0, total_g * 1.05)

    # emissions per major stage; every category keeps its slot so bars don't move
    shown_g = ts[mask].groupby("major")["emissions_kg"].sum() * 1000
    labels, values, colours, texts = [], [], [], []
    for m in majors:
        name = STAGE_COLOURS[m][1]
        partial = show is not None and m not in map(str, show) and m in shown_g.index
        if partial:  # only some stage IDs of this major stage are selected
            ids = sorted(set(ts.loc[mask & (ts["major"] == m), "stage_id"]))
            name += f" ({', '.join(ids)})"
        labels.append(name)
        if m in shown_g.index:
            values.append(shown_g[m]); colours.append(STAGE_COLOURS[m][0]); texts.append(f"{shown_g[m]:.3f} g")
        elif context == "ghost":
            values.append(by_major_full[m]); colours.append(GHOST); texts.append("")
        else:
            values.append(0); colours.append("none"); texts.append("")
    bars = ax_b.barh(labels, values, color=colours, height=0.6)
    ax_b.bar_label(bars, labels=texts, padding=4, color=INK)
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, by_major_full.max() * 1.18)
    ax_b.set_xlabel("Emissions per stage (g CO₂e)")

    for ax, g in ((ax_p, "y"), (ax_c, "y"), (ax_b, "x")):
        _style(ax, g)
    _stage_legend(fig, ts["major"])
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def reveal_steps(order, cumulative: bool = True):
    """["3", "2", "1"] -> [["3"], ["3", "2"], ["3", "2", "1"]] (or one stage per step)."""
    order = [str(o) for o in order]
    return [order[: i + 1] for i in range(len(order))] if cumulative else [[o] for o in order]


def save_reveal_sequence(ts, order, out_dir, prefix="reveal", cumulative=True, dpi=200, **plot_kwargs):
    """
    Save one overview PNG per reveal step; returns the file paths.

    ``order`` items may be stage selections as in ``plot_overview(show=...)``,
    or "all" for the whole run, e.g. ["3", "all"] gives a final-training-only
    slide followed by a whole-run slide. Pass x_range="selection" to zoom each
    slide's time axis to what it shows.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, show in enumerate(reveal_steps(order, cumulative), start=1):
        show = None if "all" in show else show
        fig = plot_overview(ts, show=show, **plot_kwargs)
        path = out_dir / f"{prefix}_{i:02d}_{'all' if show is None else '-'.join(show)}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_power_components(ts: pd.DataFrame, time_unit: str = "min", smooth: int = 1, figsize=(11, 3.5)):
    """Stacked per-interval power by hardware component (RAM is a fixed estimate)."""
    k = TIME_UNITS[time_unit]
    t = ts["elapsed_s"] / k
    parts = {c: ts[f"{c}_w"] for c in ("cpu", "gpu", "ram")}
    if smooth > 1:
        parts = {c: s.rolling(smooth, min_periods=1, center=True).mean() for c, s in parts.items()}
    fig, ax = plt.subplots(figsize=figsize)
    ax.stackplot(
        t, *parts.values(), labels=["CPU", "GPU", "RAM (fixed estimate)"],
        colors=list(COMPONENT_COLOURS.values()), step="pre", lw=0,
    )
    ax.set_xlim(0, t.max())
    ax.set_ylim(0, None)
    ax.set_ylabel("Power (W)")
    ax.set_xlabel(f"Elapsed time ({time_unit})")
    ax.set_title("Power by component", loc="left")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    _style(ax)
    fig.tight_layout()
    return fig


def plot_hpo_cost(ts: pd.DataFrame, hpo: pd.DataFrame, figsize=(7, 4.5)):
    """Emissions of each HPO trial against its validation AUC."""
    trials = stage_summary(ts).merge(hpo, on="stage_id")
    param_cols = [c for c in hpo.columns if c not in ("stage_id", "val_auc")]
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        trials["emissions_kg"] * 1000, trials["val_auc"], s=60,
        color=STAGE_COLOURS["2"][0], edgecolor="white", linewidth=2, zorder=3,
    )
    for _, r in trials.iterrows():
        ax.annotate(r["stage_id"], (r["emissions_kg"] * 1000, r["val_auc"]), xytext=(6, 4),
                    textcoords="offset points", color=INK_MUTED, fontsize=9)
    ax.set_xlabel("Trial emissions (g CO₂e)")
    ax.set_ylabel("Validation AUC")
    ax.set_title("HPO trials: cost vs performance", loc="left")
    _style(ax, "both")
    fig.tight_layout()
    return fig, trials[["stage_id", *param_cols, "val_auc", "duration_s", "energy_kwh", "emissions_kg"]]


def plot_energy_coverage(ts: pd.DataFrame, time_unit: str = "min", figsize=(11, 2.8)):
    """Fraction of wall time codecarbon counted energy for, per sample."""
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
