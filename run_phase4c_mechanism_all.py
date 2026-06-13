#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_phase4c_mechanism_all.py

One-command runner for Phase 4c mechanism analysis.

This script runs, in order:
1) phase4c_mechanism_full_log_runner.py
2) phase4c_mechanism_analysis.py

Default outputs:
~/Desktop/phase4c_mechanism_one_run/full_logs
~/Desktop/phase4c_mechanism_one_run/analysis
~/Desktop/phase4c_mechanism_one_run/run_phase4c_mechanism_all.log

Example:
cd ~/Desktop
python3 -u run_phase4c_mechanism_all.py
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from datetime import datetime


DEFAULT_CONDITIONS = (
    "full,no_self_appraisal,random_signal,label_rule_signal,"
    "no_listener_learning,no_signal_learning,private_signal"
)


class Tee:
    def __init__(self, logfile: Path):
        self.logfile = logfile
        self.fh = logfile.open("a", encoding="utf-8")

    def write(self, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()
        self.fh.write(text)
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


def resolve_path(p: str | None, default: Path) -> Path:
    if p is None or str(p).strip() == "":
        return default.expanduser().resolve()
    return Path(p).expanduser().resolve()


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found:\n  {path}\n"
            f"Place the file on Desktop or pass the correct path with the relevant argument."
        )
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file:\n  {path}")


def run_command(cmd: list[str], cwd: Path, tee: Tee) -> None:
    tee.write("\n" + "=" * 80 + "\n")
    tee.write("RUNNING:\n")
    tee.write(" ".join(shlex.quote(x) for x in cmd) + "\n")
    tee.write("=" * 80 + "\n\n")

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        tee.write(line)

    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"Command failed with exit code {code}:\n{' '.join(cmd)}")


def main() -> int:
    desktop = Path.home() / "Desktop"

    parser = argparse.ArgumentParser(
        description="Run Phase 4c mechanism full-log generation and mechanism analysis in one command."
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "quick", "full"],
        default="full",
        help="Mode passed to the full-log runner. Default: full",
    )
    parser.add_argument(
        "--core-script",
        default=str(desktop / "phase4c_self_appraisal_gap_multiagent_core_viability_fixed.py"),
        help="Path to Phase 4c fixed core script.",
    )
    parser.add_argument(
        "--log-runner",
        default=str(desktop / "phase4c_mechanism_full_log_runner.py"),
        help="Path to mechanism full-log runner script.",
    )
    parser.add_argument(
        "--analysis-script",
        default=str(desktop / "phase4c_mechanism_analysis.py"),
        help="Path to mechanism analysis script.",
    )
    parser.add_argument(
        "--outbase",
        default=str(desktop / "phase4c_mechanism_one_run"),
        help="Base output folder. Default: ~/Desktop/phase4c_mechanism_one_run",
    )
    parser.add_argument(
        "--conditions",
        default=DEFAULT_CONDITIONS,
        help="Comma-separated conditions for the full-log run.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=500,
        help="Bootstrap iterations for mechanism analysis. Default: 500",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwrite/reuse of existing output folders.",
    )

    args = parser.parse_args()

    core_script = resolve_path(args.core_script, desktop / "phase4c_self_appraisal_gap_multiagent_core_viability_fixed.py")
    log_runner = resolve_path(args.log_runner, desktop / "phase4c_mechanism_full_log_runner.py")
    analysis_script = resolve_path(args.analysis_script, desktop / "phase4c_mechanism_analysis.py")
    outbase = resolve_path(args.outbase, desktop / "phase4c_mechanism_one_run")

    require_file(core_script, "Phase 4c core script")
    require_file(log_runner, "Mechanism full-log runner script")
    require_file(analysis_script, "Mechanism analysis script")

    full_logs_dir = outbase / "full_logs"
    analysis_dir = outbase / "analysis"
    outbase.mkdir(parents=True, exist_ok=True)
    full_logs_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    log_file = outbase / "run_phase4c_mechanism_all.log"
    tee = Tee(log_file)

    try:
        tee.write("Phase 4c mechanism one-command runner\n")
        tee.write(f"Started: {datetime.now().isoformat(timespec='seconds')}\n")
        tee.write(f"Output base: {outbase}\n")
        tee.write(f"Full logs:   {full_logs_dir}\n")
        tee.write(f"Analysis:    {analysis_dir}\n")
        tee.write(f"Run log:     {log_file}\n")

        step_log = full_logs_dir / "step_logs_mechanism.csv"

        # Step 1: generate full logs.
        cmd1 = [
            sys.executable,
            "-u",
            str(log_runner),
            "--mode",
            args.mode,
            "--core-script",
            str(core_script),
            "--outdir",
            str(full_logs_dir),
            "--conditions",
            args.conditions,
        ]
        run_command(cmd1, cwd=desktop, tee=tee)

        if not step_log.exists():
            raise FileNotFoundError(
                f"Expected step log was not created:\n  {step_log}\n"
                "The full-log runner finished but the analysis input is missing."
            )

        # Step 2: mechanism analysis.
        cmd2 = [
            sys.executable,
            "-u",
            str(analysis_script),
            "--step-log",
            str(step_log),
            "--outdir",
            str(analysis_dir),
            "--bootstrap",
            str(args.bootstrap),
        ]
        run_command(cmd2, cwd=desktop, tee=tee)

        report = analysis_dir / "mechanism_analysis_report.txt"
        tee.write("\n" + "=" * 80 + "\n")
        tee.write("DONE\n")
        tee.write(f"Main report:\n  {report}\n")
        tee.write(f"Analysis folder:\n  {analysis_dir}\n")
        tee.write(f"Full-log folder:\n  {full_logs_dir}\n")
        tee.write(f"Run log:\n  {log_file}\n")
        tee.write("=" * 80 + "\n")

        if not report.exists():
            tee.write("\nWARNING: mechanism_analysis_report.txt was not found. Check the analysis folder.\n")

        return 0

    except Exception as e:
        tee.write("\nERROR\n")
        tee.write(str(e) + "\n")
        return 1
    finally:
        tee.write(f"\nFinished: {datetime.now().isoformat(timespec='seconds')}\n")
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
