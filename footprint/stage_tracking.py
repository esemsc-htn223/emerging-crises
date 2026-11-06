"""
Stage-tagged emissions timeseries on top of CodeCarbon.

CodeCarbon's built-in CSV output only writes a row on ``flush()``/``stop()``.
However, its background scheduler calls ``live_out(total, delta)`` on every
output handler every ``api_call_interval`` measurements. Registering a custom
handler with ``api_call_interval=1`` therefore gives one row per measurement
(every ``measure_power_secs`` seconds) without touching the code being measured.

Usage::

    fp = StageTracker(out_dir="outputs")
    fp.start("0.1", "imports")
    ...
    fp.stage("1.1", "download-data")      # closes 0.1, opens 1.1
    ...
    fp.stage("2.3", "hpo-trial", learning_rate=3e-4, lstm_units=64)
    ...
    fp.end()

Stage IDs follow ``<stage>.<step>``: 0 = setup/imports, 1 = data loading and
processing, 2 = hyperparameter optimisation, 3 = final training, 4 = application.

Machine vs process power
------------------------
CodeCarbon always measures (or estimates) power for the whole machine in
``tracking_mode="machine"``. Every sample additionally records this Python
process's CPU time, the whole machine's busy CPU time and the process's memory
(``proc_cpu_s``, ``sys_busy_cpu_s``, ``proc_rss_gb``, ``ram_total_gb``), so the
measured machine power can later be attributed to the process (see
``plotting.load_timeseries``). Machine minus process is the cost of everything
else running (browser tabs, editors, background services, ...).

Set the environment variable ``TRACK=0`` to turn the tracker into a no-op
(useful for checking that tracking does not affect training).
"""

from __future__ import annotations

import atexit
import csv
import json
import os
import shutil
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import psutil
from codecarbon import EmissionsTracker, OutputMethod
from codecarbon.output_methods.base_output import BaseOutput
from codecarbon.output_methods.emissions_data import EmissionsData

if TYPE_CHECKING:
    import pandas as pd

TIMESERIES_COLUMNS = [
    "timestamp",            # wall-clock time the sample was written (end of interval)
    "elapsed_s",            # seconds since tracker start (end of interval)
    "run_id",
    "stage_id",
    "stage",
    "trial_params",         # JSON, e.g. HPO trial hyperparameters
    "sample_type",          # live | boundary | final
    "duration_s",           # length of this interval
    "cpu_power_w",          # NB: codecarbon reports these three as run-to-date
    "gpu_power_w",          # averages, not per-interval power; use
    "ram_power_w",          # <part>_energy_kwh / duration_s (see plotting.py)
    "cpu_energy_kwh",       # interval values (delta) ...
    "gpu_energy_kwh",
    "ram_energy_kwh",
    "energy_kwh",
    "emissions_kg",
    "emissions_rate_kg_s",
    "cpu_util_pct",
    "ram_used_gb",
    "cum_energy_kwh",       # ... and cumulative values (total) for the run
    "cum_emissions_kg",
    "proc_cpu_s",           # cumulative CPU seconds used by this process (+ children)
    "sys_busy_cpu_s",       # cumulative busy CPU seconds of the whole machine
    "proc_rss_gb",          # resident memory of this process (+ children)
    "ram_total_gb",         # installed RAM
    "tracking_mode",        # codecarbon tracking mode: machine | process
]

TrackingMode = Literal["machine", "process"]
SampleType = Literal["live", "boundary", "final"]


def _process_snapshot() -> dict[str, float]:
    """
    Read cumulative CPU-time and memory counters for attribution.

    Returns
    -------
    dict of str to float
        ``proc_cpu_s``: user + system CPU seconds of this process and its
        children (live and already-exited). ``sys_busy_cpu_s``: busy (non-idle)
        CPU seconds summed over all cores of the machine. ``proc_rss_gb``:
        resident memory of this process and its live children, in GB.
    """
    proc = psutil.Process()
    times = proc.cpu_times()
    proc_cpu = times.user + times.system + times.children_user + times.children_system
    rss = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            child_times = child.cpu_times()
            proc_cpu += child_times.user + child_times.system
            rss += child.memory_info().rss
        except psutil.Error:  # child exited between listing and reading
            continue
    sys_times = psutil.cpu_times()
    idle = sys_times.idle + getattr(sys_times, "iowait", 0.0)
    sys_busy = sum(sys_times) - idle
    return {"proc_cpu_s": proc_cpu, "sys_busy_cpu_s": sys_busy, "proc_rss_gb": rss / 1024**3}


def _migrate_header(path: Path, columns: list[str]) -> None:
    """
    Extend an existing timeseries CSV to the current column set, in place.

    Older files lack newer columns (e.g. the process-attribution ones).
    Appending new rows to them would misalign every value, so the file is
    rewritten with the full header and blank values for the missing columns.
    A copy of the original is kept as ``<name>.bak-<timestamp>``.

    Parameters
    ----------
    path : Path
        Existing timeseries CSV.
    columns : list of str
        The column set rows will be written with.

    Raises
    ------
    ValueError
        If the file has columns that ``columns`` does not contain, which would
        mean data loss on rewrite.
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        old_columns = reader.fieldnames or []
        if old_columns == columns:
            return
        rows = list(reader)
    unknown = set(old_columns) - set(columns)
    if unknown:
        raise ValueError(f"{path} has unexpected columns {sorted(unknown)}; not rewriting it.")
    backup = path.with_name(f"{path.name}.bak-{datetime.now():%Y%m%d-%H%M%S}")
    shutil.copy2(path, backup)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[footprint] extended {path.name} with new columns (backup: {backup.name})")


class StageTimeseriesOutput(BaseOutput):
    """
    CodeCarbon output handler writing one CSV row per measurement, tagged by stage.

    Parameters
    ----------
    path : str or Path
        CSV file to append to. Created with a header if missing; an existing
        file with an older column set is extended in place (with a backup).
    tracking_mode : {"machine", "process"}, default "machine"
        CodeCarbon tracking mode, recorded on every row.

    Attributes
    ----------
    stage_id, stage : str
        Labels written on each row until changed with :meth:`set_stage`.
    closing : {"boundary", "final"} or None
        Set by :class:`StageTracker` while it closes a stage, so rows produced
        by ``flush()``/``stop()`` are labelled as such rather than "live".
    """

    MIN_DURATION_S = 1e-3  # skip the zero-length duplicate rows that flush() can produce

    def __init__(self, path: str | Path, tracking_mode: TrackingMode = "machine") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tracking_mode = tracking_mode
        self._lock = threading.Lock()
        self.stage_id = "-"
        self.stage = "untagged"
        self.trial_params = ""
        self.closing: SampleType | None = None
        self._ram_total_gb = psutil.virtual_memory().total / 1024**3
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        if not new_file:
            _migrate_header(self.path, TIMESERIES_COLUMNS)
        self._fh = open(self.path, "a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=TIMESERIES_COLUMNS)
        if new_file:
            self._writer.writeheader()
            self._fh.flush()

    def set_stage(self, stage_id: str, stage: str, trial_params: dict[str, Any] | None = None) -> None:
        """
        Change the labels written on subsequent rows.

        Parameters
        ----------
        stage_id : str
            Stage ID such as ``"2.3"``.
        stage : str
            Human-readable stage name such as ``"hpo-trial"``.
        trial_params : dict, optional
            Extra parameters stored as JSON in the ``trial_params`` column.
        """
        with self._lock:
            self.stage_id = str(stage_id)
            self.stage = stage
            self.trial_params = json.dumps(trial_params, default=str) if trial_params else ""

    def _write(self, total: EmissionsData, delta: EmissionsData, sample_type: SampleType) -> None:
        """
        Append one row for the interval described by ``delta``.

        Parameters
        ----------
        total : EmissionsData
            Run-to-date totals from CodeCarbon.
        delta : EmissionsData
            Values for the interval since the previous row.
        sample_type : {"live", "boundary", "final"}
            Row label; overridden by :attr:`closing` when it is set.
        """
        if delta.duration < self.MIN_DURATION_S:
            return
        sample_type = self.closing or sample_type
        with self._lock:
            if self._fh.closed:
                return
            self._writer.writerow({
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "elapsed_s": total.duration,
                "run_id": total.run_id,
                "stage_id": self.stage_id,
                "stage": self.stage,
                "trial_params": self.trial_params,
                "sample_type": sample_type,
                "duration_s": delta.duration,
                "cpu_power_w": delta.cpu_power,
                "gpu_power_w": delta.gpu_power,
                "ram_power_w": delta.ram_power,
                "cpu_energy_kwh": delta.cpu_energy,
                "gpu_energy_kwh": delta.gpu_energy,
                "ram_energy_kwh": delta.ram_energy,
                "energy_kwh": delta.energy_consumed,
                "emissions_kg": delta.emissions,
                "emissions_rate_kg_s": delta.emissions_rate,
                "cpu_util_pct": delta.cpu_utilization_percent,
                "ram_used_gb": delta.ram_used_gb,
                "cum_energy_kwh": total.energy_consumed,
                "cum_emissions_kg": total.emissions,
                **_process_snapshot(),
                "ram_total_gb": self._ram_total_gb,
                "tracking_mode": self.tracking_mode,
            })
            self._fh.flush()

    def live_out(self, total: EmissionsData, delta: EmissionsData) -> None:
        """
        Write a row; called by CodeCarbon's scheduler after every measurement.

        Parameters
        ----------
        total : EmissionsData
            Run-to-date totals.
        delta : EmissionsData
            Values since the previous row.
        """
        self._write(total, delta, "live")

    def out(self, total: EmissionsData, delta: EmissionsData) -> None:
        """
        Write a row; called by CodeCarbon on ``flush()`` and ``stop()``.

        Parameters
        ----------
        total : EmissionsData
            Run-to-date totals.
        delta : EmissionsData
            Values since the previous row.
        """
        self._write(total, delta, "boundary")

    def exit(self) -> None:
        """Close the CSV file; called by CodeCarbon on ``stop()``."""
        with self._lock:
            if not self._fh.closed:
                self._fh.close()


@contextmanager
def _prefer_powermetrics(enabled: bool = True) -> Iterator[None]:
    """
    Work around codecarbon 3.3.2 never using Apple PowerMetrics on Apple Silicon.

    ``ResourceTracker._try_platform_cpu_backend`` tries the psutil-based
    "cpu_load" estimate *before* PowerMetrics, and that estimate always
    succeeds, so the PowerMetrics branch is unreachable even when passwordless
    ``sudo powermetrics`` is configured. If PowerMetrics is usable, skip the
    estimate for the duration of hardware detection so codecarbon's own
    PowerMetrics backend (measured CPU + GPU power) is selected instead.
    Without passwordless sudo this does nothing.

    Parameters
    ----------
    enabled : bool, default True
        Set False to leave codecarbon's backend choice untouched.

    Yields
    ------
    None
        Hardware detection should run inside the ``with`` block.
    """
    from codecarbon.core import powermetrics
    from codecarbon.core.resource_tracker import ResourceTracker
    from codecarbon.core.util import is_mac_os

    if not (enabled and is_mac_os() and powermetrics.is_powermetrics_available()):
        yield
        return
    original = ResourceTracker._setup_cpu_load_fast
    ResourceTracker._setup_cpu_load_fast = lambda self, model: False
    try:
        yield
    finally:
        ResourceTracker._setup_cpu_load_fast = original


def _count_measurement_time(tracker: EmissionsTracker) -> None:
    """
    Work around codecarbon 3.3.2 dropping the time spent taking measurements.

    ``_do_measurements`` computes each component's energy as its power reading
    times the time since the *previous measurement finished*
    (``_last_measured_time`` is only updated after every component has been
    read). Any time spent inside the measurement itself is therefore never
    counted. That is negligible for fast backends, but each Apple PowerMetrics
    reading runs ``sudo powermetrics`` for ~1.2 s (once for the CPU, once for
    the GPU), so only ~50% of wall time received any energy.

    Fix: give every hardware component its own clock. Each reading's energy
    covers the time since *that component's* previous reading, so a
    component's readings tile the run with no gaps and no overlaps, even when
    the scheduler thread and ``flush()`` measure concurrently. The first
    reading keeps codecarbon's value (time since the tracker started).

    Parameters
    ----------
    tracker : EmissionsTracker
        Tracker whose hardware has been detected but which has not started.
    """
    for hardware in tracker._hardware:
        original = hardware.measure_power_and_energy
        state: dict[str, Any] = {"prev": None, "lock": threading.Lock()}

        def measure(*args: Any, last_duration: float, _orig=original, _state=state, **kwargs: Any) -> Any:
            now = time.perf_counter()
            with _state["lock"]:
                if _state["prev"] is not None:
                    last_duration = now - _state["prev"]
                _state["prev"] = now
            return _orig(*args, last_duration=last_duration, **kwargs)

        hardware.measure_power_and_energy = measure


class StageTracker:
    """
    One CodeCarbon tracker for a whole workflow, with switchable stage labels.

    Parameters
    ----------
    out_dir : str or Path, default "outputs"
        Folder for the timeseries and summary CSVs.
    timeseries_file : str, default "emissions_timeseries.csv"
        One row per sample, appended across runs (rows carry ``run_id``).
    summary_file : str, default "emissions_summary.csv"
        CodeCarbon's own one-row-per-run summary.
    measure_power_secs : float, default 5
        Sampling interval in seconds.
    project_name, experiment_name : str
        Passed to CodeCarbon.
    tracking_mode : {"machine", "process"}, optional
        CodeCarbon's tracking mode. Defaults to the ``FOOTPRINT_TRACKING_MODE``
        environment variable, else ``"machine"``. ``"process"`` only changes
        CodeCarbon's *estimates* (CPU power from the process's CPU load, RAM
        from its memory) and is ignored by measured backends such as Apple
        PowerMetrics, so PowerMetrics is not used in process mode. Machine
        mode plus the recorded attribution columns is usually the better way
        to get process power.
    enabled : bool, optional
        Defaults to the ``TRACK`` environment variable (off for 0/false/no).
        When disabled every method is a no-op.
    prefer_powermetrics : bool, default True
        Use Apple PowerMetrics when available (see ``_prefer_powermetrics``).
    count_measurement_time : bool, default True
        Apply the measurement-time fix (see ``_count_measurement_time``).
    **tracker_kwargs
        Extra keyword arguments for ``codecarbon.EmissionsTracker``, e.g.
        ``force_carbon_intensity_g_co2e_kwh`` or ``pue``.

    Attributes
    ----------
    run_id : str
        CodeCarbon run ID, written on every row.
    power_source : str
        Hardware CodeCarbon is reading, set by :meth:`start`.
    """

    def __init__(
        self,
        out_dir: str | Path = "outputs",
        timeseries_file: str = "emissions_timeseries.csv",
        summary_file: str = "emissions_summary.csv",
        measure_power_secs: float = 5,
        project_name: str = "ml-footprint",
        experiment_name: str = "lstm-workflow",
        tracking_mode: TrackingMode | None = None,
        enabled: bool | None = None,
        prefer_powermetrics: bool = True,
        count_measurement_time: bool = True,
        **tracker_kwargs: Any,
    ) -> None:
        if enabled is None:
            enabled = os.getenv("TRACK", "1").lower() not in ("0", "false", "no")
        if tracking_mode is None:
            tracking_mode = os.getenv("FOOTPRINT_TRACKING_MODE", "machine")  # type: ignore[assignment]
        if tracking_mode not in ("machine", "process"):
            raise ValueError(f"tracking_mode must be 'machine' or 'process', not {tracking_mode!r}")
        self.enabled = enabled
        self.tracking_mode: TrackingMode = tracking_mode
        self.out_dir = Path(out_dir)
        self.timeseries_path = self.out_dir / timeseries_file
        self.summary_path = self.out_dir / summary_file
        self.current: tuple[str, str] | None = None
        # measured backends ignore process mode, so fall back to estimates there
        self.prefer_powermetrics = prefer_powermetrics and tracking_mode == "machine"
        self.count_measurement_time = count_measurement_time
        self.power_source = "disabled"
        self._ended = False
        if not self.enabled:
            print("StageTracker disabled (TRACK=0): no emissions are recorded.")
            return

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.handler = StageTimeseriesOutput(self.timeseries_path, tracking_mode=tracking_mode)
        self.tracker = EmissionsTracker(
            project_name=project_name,
            experiment_name=experiment_name,
            measure_power_secs=measure_power_secs,
            api_call_interval=1,            # live_out after every measurement
            output_dir=str(self.out_dir),
            output_file=summary_file,
            output_methods=[OutputMethod.CSV],
            on_csv_write="update",          # one summary row per run_id
            output_handlers=[self.handler],
            tracking_mode=tracking_mode,
            log_level="warning",
            **tracker_kwargs,
        )
        self.run_id = str(self.tracker.run_id)

    def start(self, stage_id: str, stage: str, **trial_params: Any) -> None:
        """
        Start the tracker and open the first stage.

        Parameters
        ----------
        stage_id : str
            First stage ID, e.g. ``"0.1"``.
        stage : str
            Its name, e.g. ``"imports"``.
        **trial_params
            Stored as JSON on this stage's rows.
        """
        self.current = (str(stage_id), stage)
        if not self.enabled:
            return
        self.handler.set_stage(stage_id, stage, trial_params)
        # hardware detection normally happens lazily inside start(); do it first
        # so the measurement-time fix is in place before the first reading
        with _prefer_powermetrics(self.prefer_powermetrics):
            self.tracker._ensure_hardware_ready()
        if self.count_measurement_time:
            _count_measurement_time(self.tracker)
        self.tracker.start()
        atexit.register(self.end)
        self.power_source = ", ".join(self.tracker._conf.get("hardware", [])) or "unknown"
        print(f"[footprint] run {self.run_id}: started stage {stage_id} ({stage})")
        print(f"[footprint] tracking mode: {self.tracking_mode}; power source: {self.power_source}")

    def stage(self, stage_id: str, stage: str, **trial_params: Any) -> None:
        """
        Close the current stage at this exact moment and open a new one.

        ``flush()`` measures up to now and writes that interval under the
        outgoing label, so stage boundaries are exact rather than rounded to
        the sampling grid.

        Parameters
        ----------
        stage_id : str
            New stage ID, e.g. ``"2.3"``.
        stage : str
            Its name, e.g. ``"hpo-trial"``.
        **trial_params
            Stored as JSON on this stage's rows, e.g. hyperparameters.
        """
        self.current = (str(stage_id), stage)
        if not self.enabled:
            return
        self.handler.closing = "boundary"
        try:
            self.tracker.flush()
        finally:
            self.handler.closing = None
        self.handler.set_stage(stage_id, stage, trial_params)

    @contextmanager
    def stage_ctx(self, stage_id: str, stage: str, **trial_params: Any) -> Iterator[StageTracker]:
        """
        Context-manager form of :meth:`stage`.

        Parameters
        ----------
        stage_id, stage, **trial_params
            As for :meth:`stage`.

        Yields
        ------
        StageTracker
            This tracker.

        Examples
        --------
        >>> with fp.stage_ctx("2.1", "hpo-trial", lr=1e-3):  # doctest: +SKIP
        ...     model.fit(X, y)
        """
        self.stage(stage_id, stage, **trial_params)
        yield self

    def end(self) -> None:
        """Stop the tracker, writing the last partial interval of the current stage."""
        if not self.enabled or self._ended:
            return
        self._ended = True
        self.handler.closing = "final"
        self.tracker.stop()
        print(f"[footprint] run {self.run_id}: stopped. Timeseries -> {self.timeseries_path}")

    def timeseries(self, this_run_only: bool = True) -> pd.DataFrame:
        """
        Read the raw timeseries CSV.

        Parameters
        ----------
        this_run_only : bool, default True
            Keep only this tracker's rows.

        Returns
        -------
        pandas.DataFrame
            Raw rows; use ``plotting.load_timeseries`` for derived columns.
        """
        import pandas as pd

        df = pd.read_csv(self.timeseries_path, dtype={"stage_id": str})
        if this_run_only and self.enabled:
            df = df[df["run_id"] == self.run_id]
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.reset_index(drop=True)

    def summary(self, this_run_only: bool = True) -> pd.DataFrame:
        """
        Energy, emissions and duration totals per stage.

        Parameters
        ----------
        this_run_only : bool, default True
            Keep only this tracker's rows.

        Returns
        -------
        pandas.DataFrame
            One row per stage ID.
        """
        df = self.timeseries(this_run_only)
        df["major_stage"] = df["stage_id"].str.split(".").str[0]
        out = (
            df.groupby(["stage_id", "stage"], sort=False)
            .agg(
                major_stage=("major_stage", "first"),
                duration_s=("duration_s", "sum"),
                n_samples=("duration_s", "size"),
                energy_kwh=("energy_kwh", "sum"),
                emissions_kg=("emissions_kg", "sum"),
                cpu_energy_kwh=("cpu_energy_kwh", "sum"),
            )
            .reset_index()
        )
        out["mean_cpu_power_w"] = out.pop("cpu_energy_kwh") * 3.6e6 / out["duration_s"]
        out["share_of_emissions_pct"] = 100 * out["emissions_kg"] / out["emissions_kg"].sum()
        return out
