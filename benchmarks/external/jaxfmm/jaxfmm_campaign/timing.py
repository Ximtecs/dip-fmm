# SPDX-License-Identifier: Apache-2.0
"""Timing statistics shared by both runners.

One *sample* times ``evaluations`` consecutive field updates and reports the
mean per update; the headline is the median over samples (upper median for
an even count, as in the FMM3D campaign), with min, max and mean retained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TimingProtocol:
    warmups: int = 3
    samples: int = 5
    evaluations: int = 5

    def as_dict(self) -> dict[str, int]:
        return {"warmups": self.warmups, "samples": self.samples, "evaluations": self.evaluations}


@dataclass
class TimingSummary:
    per_sample_seconds: list[float] = field(default_factory=list)

    @property
    def median(self) -> float:
        ordered = sorted(self.per_sample_seconds)
        return ordered[len(ordered) // 2] if ordered else float("nan")

    def as_dict(self, prefix: str) -> dict[str, Any]:
        values = self.per_sample_seconds
        if not values:
            return {f"{prefix}_median_seconds": None}
        return {
            f"{prefix}_median_seconds": self.median,
            f"{prefix}_min_seconds": min(values),
            f"{prefix}_max_seconds": max(values),
            f"{prefix}_mean_seconds": sum(values) / len(values),
            f"{prefix}_samples_seconds": list(values),
        }
