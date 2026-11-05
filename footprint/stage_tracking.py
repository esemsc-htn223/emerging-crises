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

Set the environment variable ``TRACK=0`` to turn the tracker into a no-op
(useful for checking that tracking does not affect training).
"""

from __future__ import annotations

import atexit
import csv
import json
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from codecarbon import EmissionsTracker, OutputMethod
from codecarbon.output_methods.base_output import BaseOutput
from codecarbon.output_methods.emissions_data import EmissionsData

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
]


class StageTimeseriesOutput(BaseOutput):
    """CodeCarbon output handler writing one CSV row per measurement, tagged by stage."""

    MIN_DURATION_S = 1e-3  # skip the zero-length duplicate rows that flush() can produce

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.stage_id = "-"
        self.stage = "untagged"
        self.trial_params = ""
        # set while StageTracker closes a stage (flush/stop) so those rows are
        # labelled "boundary"/"final" rather than "live"
        self.closing: str | None = None
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        self._fh = open(self.path, "a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=TIMESERIES_COLUMNS)
        if new_file:
            self._writer.writeheader()
            self._fh.flush()

    def set_stage(self, stage_id: str, stage: str, trial_params: dict | None = None):
        with self._lock:
            self.stage_id = str(stage_id)
            self.stage = stage
            self.trial_params = json.dumps(trial_params, default=str) if trial_params else ""

    def _write(self, total: EmissionsData, delta: EmissionsData, sample_type: str):
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
            })
            self._fh.flush()

    # called by the background scheduler every measurement
    def live_out(self, total: EmissionsData, delta: EmissionsData):
        self._write(total, delta, "live")

    # called by tracker.flush() / tracker.stop()
    def out(self, total: EmissionsData, delta: EmissionsData):
        self._write(total, delta, "boundary")

    def exit(self):
        with self._lock:
            if not self._fh.closed:
                self._fh.close()


@contextmanager
def _prefer_powermetrics(enabled: bool = True):
    """
    Work around codecarbon 3.3.2 never using Apple PowerMetrics on Apple Silicon.

    ``ResourceTracker._try_platform_cpu_backend`` tries the psutil-based
    "cpu_load" estimate *before* PowerMetrics, and that estimate always
    succeeds, so the PowerMetrics branch is unreachable even when passwordless
    ``sudo powermetrics`` is configured. If PowerMetrics is usable, skip the
    estimate for the duration of hardware detection so codecarbon's own
    PowerMetrics backend (measured CPU + GPU power) is selected instead.
    Without passwordless sudo this does nothing.
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
    Must be applied after hardware detection and before ``tracker.start()``.
    """
    for hardware in tracker._hardware:
        original = hardware.measure_power_and_energy
        state = {"prev": None, "lock": threading.Lock()}

        def measure(*args, last_duration, _orig=original, _state=state, **kwargs):
            now = time.perf_counter()
            with _state["lock"]:
                if _state["prev"] is not None:
                    last_duration = now - _state["prev"]
                _state["prev"] = now
            return _orig(*args, last_duration=last_duration, **kwargs)

        hardware.measure_power_and_energy = measure


class StageTracker:
    """One CodeCarbon tracker for a whole workflow, with switchable stage labels."""

    def __init__(
        self,
        out_dir: str | Path = "outputs",
        timeseries_file: str = "emissions_timeseries.csv",
        summary_file: str = "emissions_summary.csv",
        measure_power_secs: float = 5,
        project_name: str = "ml-footprint",
        experiment_name: str = "lstm-workflow",
        enabled: bool | None = None,
        prefer_powermetrics: bool = True,
        count_measurement_time: bool = True,
        **tracker_kwargs: Any,
    ):
        if enabled is None:
            enabled = os.getenv("TRACK", "1").lower() not in ("0", "false", "no")
        self.enabled = enabled
        self.out_dir = Path(out_dir)
        self.timeseries_path = self.out_dir / timeseries_file
        self.summary_path = self.out_dir / summary_file
        self.current: tuple[str, str] | None = None
        self.prefer_powermetrics = prefer_powermetrics
        self.count_measurement_time = count_measurement_time
        self.power_source = "disabled"
        self._ended = False
        if not self.enabled:
            print("StageTracker disabled (TRACK=0): no emissions are recorded.")
            return

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.handler = StageTimeseriesOutput(self.timeseries_path)
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
            tracking_mode="machine",
            log_level="warning",
            **tracker_kwargs,
        )
        self.run_id = str(self.tracker.run_id)

    def start(self, stage_id: str, stage: str, **trial_params):
        """Start the tracker and open the first stage."""
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
        print(f"[footprint] power source: {self.power_source}")

    def stage(self, stage_id: str, stage: str, **trial_params):
        """Close the current stage (writing its final partial interval) and open a new one."""
        self.current = (str(stage_id), stage)
        if not self.enabled:
            return
        # flush() measures up to *now* and writes it under the outgoing label,
        # so stage boundaries are exact rather than rounded to the sampling grid.
        self.handler.closing = "boundary"
        try:
            self.tracker.flush()
        finally:
            self.handler.closing = None
        self.handler.set_stage(stage_id, stage, trial_params)

    @contextmanager
    def stage_ctx(self, stage_id: str, stage: str, **trial_params):
        """``with fp.stage_ctx("2.1", "hpo-trial", lr=1e-3): ...`` (same as ``stage``)."""
        self.stage(stage_id, stage, **trial_params)
        yield self

    def end(self):
        """Stop the tracker, writing the last partial interval of the current stage."""
        if not self.enabled or self._ended:
            return
        self._ended = True
        self.handler.closing = "final"
        self.tracker.stop()
        print(f"[footprint] run {self.run_id}: stopped. Timeseries -> {self.timeseries_path}")

    def timeseries(self, this_run_only: bool = True):
        import pandas as pd

        df = pd.read_csv(self.timeseries_path, dtype={"stage_id": str})
        if this_run_only and self.enabled:
            df = df[df["run_id"] == self.run_id]
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.reset_index(drop=True)

    def summary(self, this_run_only: bool = True):
        """Energy, emissions and duration totals per stage."""
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
