#!/usr/bin/env python3
"""Quick Zarr attribute inspector (debug utility)."""
from __future__ import annotations

import zarr


if __name__ == "__main__":
    z_lig = zarr.open("results/ref_S1P_fixed_new2/data/lig_sensitivities.zarr", mode="r")
    print("Ligament names:", z_lig.attrs.get("ligament_names", "Not found"))
