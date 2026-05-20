"""Parameter sweep framework for scientific simulations.

Hash-based directory naming ensures uniqueness across all parameter combinations.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Protocol

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from logging_config import get_logger

ParamValue = int | float | str | bool | None
ParamDict = Dict[str, List[ParamValue]]


def _relative_str(path: Path | str, base: Path | None) -> str:
    p = Path(path)
    if base is not None:
        base = Path(base)
        if p == base or base in p.parents:
            return str(p.relative_to(base))
    return str(p)


class Runnable(Protocol):
    def __call__(self, param_point: Dict[str, ParamValue], output_path: Path) -> None: ...


@dataclass
class ParameterSweep:
    """Parameter sweep configuration with hash-based output naming."""
    params: ParamDict
    base_output_dir: Path
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.params:
            raise ValueError("params dictionary cannot be empty")
        
        for key, values in self.params.items():
            if not isinstance(values, list):
                self.params[key] = [values]
        
        self.base_output_dir = Path(self.base_output_dir)
    
    def generate_points(self) -> List[Dict[str, ParamValue]]:
        """Generate all parameter combinations (Cartesian product)."""
        param_names = list(self.params.keys())
        param_values = [self.params[k] for k in param_names]
        return [dict(zip(param_names, values)) for values in itertools.product(*param_values)]
    
    def format_output_path(self, param_point: Dict[str, ParamValue]) -> Path:
        """Generate output path using 8-char hash of parameter point."""
        param_str = json.dumps(param_point, sort_keys=True, separators=(",", ":"))
        params_hash = hashlib.sha1(param_str.encode("utf-8")).hexdigest()[:8]
        return self.base_output_dir / params_hash
    
    def total_runs(self) -> int:
        """Calculate total number of runs in sweep."""
        total = 1
        for values in self.params.values():
            total *= len(values)
        return total


@dataclass
class SweepResult:
    """Results from a completed parameter sweep."""
    param_points: List[Dict[str, ParamValue]]
    output_paths: List[Path]
    metadata: Dict[str, Any] = field(default_factory=dict)
    base_output_dir: Path | None = None
    
    def save_json(self, output_file: Path) -> None:
        """Save sweep results to JSON."""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        summary = {
            "total_runs": len(self.param_points),
            "runs": [
                {
                    "parameters": params,
                    "output_path": _relative_str(path, self.base_output_dir)
                }
                for params, path in zip(self.param_points, self.output_paths)
            ]
        }
        
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
    
    def save_csv(self, output_file: Path) -> None:
        """Save sweep results to CSV."""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.param_points:
            return
        
        param_keys = set()
        for params in self.param_points:
            param_keys.update(params.keys())
        
        header = sorted(param_keys) + ["output_path"]
        
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
            writer.writeheader()
            for params, path in zip(self.param_points, self.output_paths):
                row = {**params, "output_path": _relative_str(path, self.base_output_dir)}
                writer.writerow(row)


class Parametrizer:
    """Execute parameter sweeps with arbitrary callables."""
    
    def __init__(
        self,
        sweep: ParameterSweep,
        callable_func: Runnable,
        verbose: bool = True
    ):
        self.sweep = sweep
        self.callable_func = callable_func
        self.logger = get_logger("Parametrizer")
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)
        self._show_progress = bool(verbose)
    
    def run(self) -> SweepResult:
        """Execute parameter sweep."""
        param_points = self.sweep.generate_points()
        total_runs = len(param_points)
        
        self.logger.info(f"Parameter sweep: {total_runs} runs")
        self.sweep.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        output_paths: List[Path] = []
        
        iterator = enumerate(param_points)
        if self._show_progress:
            iterator = tqdm(iterator, total=total_runs, desc="Parameter Sweep", unit="run", ncols=100)
        
        for idx, param_point in iterator:
            output_path = self.sweep.format_output_path(param_point)
            self.callable_func(param_point, output_path)
            output_paths.append(output_path)
        
        result = SweepResult(
            param_points=param_points,
            output_paths=output_paths,
            metadata={**self.sweep.metadata, "total_runs": total_runs},
            base_output_dir=self.sweep.base_output_dir,
        )
        
        csv_file = self.sweep.base_output_dir / "sweep_results.csv"
        json_file = self.sweep.base_output_dir / "sweep_summary.json"
        
        result.save_csv(csv_file)
        result.save_json(json_file)
        
        self.logger.info(f"Sweep complete. Results: {csv_file}")
        
        return result
    
    def run_single(self, param_point: Dict[str, ParamValue], output_path: Path | None = None) -> None:
        """Execute single run with given parameters."""
        if output_path is None:
            output_path = self.sweep.format_output_path(param_point)
        self.callable_func(param_point, output_path)
