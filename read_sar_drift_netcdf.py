# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 10:32:13 2026

netcdf_inspect_export_plot.py

Usage:
  python netcdf_inspect_export_plot.py /path/to/file.nc --out ./out --max-slices 50

What it does:
  1) Shows properties
  2) Shows layers (data variables)
  3) Exports each layer (and its slices) to NumPy arrays (.npy)
  4) Plots each slice (2D)
  5) Saves each plot as PNG
"""

# from __future__ import annotations

# import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def safe_name(s: str) -> str:
    """Make a string safe for filenames."""
    s = str(s)
    s = re.sub(r"[^\w\-\.]+", "_", s)
    return s.strip("_")[:180]  # keep filenames manageable


def print_dataset_properties(ds: xr.Dataset) -> None:
    print("\n=== DATASET SUMMARY ===")
    print(ds)

    print("\n=== GLOBAL ATTRIBUTES ===")
    if ds.attrs:
        for k, v in ds.attrs.items():
            print(f"- {k}: {v}")
    else:
        print("(none)")

    print("\n=== DIMS ===")
    for d, n in ds.dims.items():
        print(f"- {d}: {n}")

    print("\n=== COORDS ===")
    if ds.coords:
        for c in ds.coords:
            da = ds.coords[c]
            print(f"- {c}: dims={da.dims}, dtype={da.dtype}, shape={da.shape}")
    else:
        print("(none)")


def list_layers(ds: xr.Dataset) -> None:
    print("\n=== DATA VARIABLES (LAYERS) ===")
    if not ds.data_vars:
        print("(none)")
        return

    for v in ds.data_vars:
        da = ds[v]
        units = da.attrs.get("units", "")
        long_name = da.attrs.get("long_name", da.attrs.get("standard_name", ""))
        print(f"\n- {v}")
        print(f"  dims: {da.dims}")
        print(f"  shape: {da.shape}")
        print(f"  dtype: {da.dtype}")
        if long_name:
            print(f"  name: {long_name}")
        if units:
            print(f"  units: {units}")
        if da.attrs:
            # show a few key attrs without spamming
            keys = [k for k in da.attrs.keys() if k not in ("units", "long_name", "standard_name")]
            if keys:
                show = keys[:8]
                print(f"  attrs: {', '.join(show)}" + (" ..." if len(keys) > len(show) else ""))


def choose_2d_dims(da: xr.DataArray) -> Optional[Tuple[str, str]]:
    """
    Pick which two dimensions to plot as a 2D image.
    Preference: (y,x), (lat,lon), (latitude,longitude), otherwise last two dims.
    """
    dims = list(da.dims)
    if len(dims) < 2:
        return None

    preferred_pairs = [
        ("y", "x"),
        ("lat", "lon"),
        ("latitude", "longitude"),
        ("nav_lat", "nav_lon"),
        ("rows", "cols"),
    ]
    for a, b in preferred_pairs:
        if a in dims and b in dims:
            return a, b

    # fallback: last two dims
    return dims[-2], dims[-1]


def iter_2d_slices(
    da: xr.DataArray,
    xy_dims: Tuple[str, str],
    max_slices: int,
) -> List[Tuple[Dict[str, int], xr.DataArray]]:
    """
    For a DataArray with dims possibly >2, generate 2D slices by indexing
    all non-(x,y) dims.
    Returns list of (indexers, slice_da).
    """
    xdim, ydim = xy_dims[1], xy_dims[0]  # keep consistent naming? We'll treat as (y,x) for image
    plot_dims = set([xy_dims[0], xy_dims[1]])
    extra_dims = [d for d in da.dims if d not in plot_dims]

    if not extra_dims:
        return [({}, da.transpose(xy_dims[0], xy_dims[1]))]

    # Build index combinations for extra dims
    combos: List[Dict[str, int]] = [{}]
    for d in extra_dims:
        new_combos = []
        for base in combos:
            for i in range(da.sizes[d]):
                dd = dict(base)
                dd[d] = i
                new_combos.append(dd)
        combos = new_combos

    # limit
    combos = combos[:max_slices]

    out: List[Tuple[Dict[str, int], xr.DataArray]] = []
    for idx in combos:
        sl = da.isel(**idx).transpose(xy_dims[0], xy_dims[1])
        out.append((idx, sl))
    return out


def export_array(out_dir: Path, varname: str, idx: Dict[str, int], arr: np.ndarray) -> Path:
    """Save array as .npy and return the file path."""
    idx_part = "__".join([f"{k}{v}" for k, v in idx.items()]) if idx else "full"
    fname = f"{safe_name(varname)}__{safe_name(idx_part)}.npy"

    arrays_dir = os.path.join(str(out_dir), "arrays")
    os.makedirs(arrays_dir, exist_ok=True)          # makedirs supports exist_ok

    fpath = os.path.join(arrays_dir, fname)
    np.save(fpath, arr)
    return Path(fpath)


def plot_and_save_png(
    out_dir: Path,
    varname: str,
    idx: Dict[str, int],
    slice_da: xr.DataArray,
    robust: bool = True,
) -> Path:
    """
    Plot a 2D slice and save to PNG.
    Uses imshow (no map projection). Works for any 2D grid.
    """
    idx_part = "__".join([f"{k}{v}" for k, v in idx.items()]) if idx else "full"
    title = f"{varname} [{idx_part}]"
    units = slice_da.attrs.get("units", "")

    data = slice_da.values
    # handle masked/NaNs gracefully
    data = np.array(data)

    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    im = ax.imshow(data, origin="lower", aspect="auto")

    ax.set_title(f"{title}" + (f" ({units})" if units else ""))
    ax.set_xlabel(slice_da.dims[1])
    ax.set_ylabel(slice_da.dims[0])
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    if units:
        cbar.set_label(units)

    png_dir = os.path.join(out_dir, "png")
    os.makedirs(png_dir, exist_ok=True)
    
    png_name = f"{safe_name(varname)}__{safe_name(idx_part)}.png"
    png_path = os.path.join(png_dir, png_name)
    

    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)
    return png_path


def plot_netcdf(ds, base_name, output_dir, step=8):
    # 2D slices (time=0)
    dx = ds["dx"].isel(time=0)
    dy = ds["dy"].isel(time=0)
    
    # Coordinates (usually x/y in meters)
    x = ds["x"].values
    y = ds["y"].values
    

    X, Y = np.meshgrid(x[::step], y[::step])
    u = dx.values[::step, ::step]
    v = dy.values[::step, ::step]
    
    # Mask invalids
    mask = np.isfinite(u) & np.isfinite(v)
    u = np.where(mask, u, np.nan)
    v = np.where(mask, v, np.nan)
    mag = np.hypot(u, v)
    
    fig, ax = plt.subplots(figsize=(10, 9), dpi=150)

    
    # Arrows
    q = ax.quiver(
         X, Y, u, v, mag,
         angles="xy", scale_units="xy", scale=.05,
         width=0.0025, cmap='viridis'
    )
    cbar = fig.colorbar(q, ax=ax, shrink=0.85)
    cbar.set_label("Distance m/day")
    
    ax.set_title("Ice drift vectors (u, v) [EPSG:3413]")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.ticklabel_format(axis="both", style="plain", useOffset=False)
    
    png_dir = os.path.join(output_dir, "png")
    os.makedirs(png_dir, exist_ok=True)
    
    png_name = f"{base_name}.png"
    png_path = os.path.join(png_dir, png_name)
    

    fig.tight_layout()
    fig.savefig(png_path)
    plt.show()
    plt.close(fig)
    
    
    

def main():
    # p = argparse.ArgumentParser()
    # p.add_argument("nc_path", type=str, help="Path to NetCDF file (.nc)")
    # p.add_argument("--out", type=str, default="./netcdf_out", help="Output folder")
    # p.add_argument("--max-slices", type=int, default=40, help="Max 2D slices per variable")
    # p.add_argument("--decode-times", action="store_true", help="Decode CF times (may be slower)")
    # args = p.parse_args()

    # nc_path = Path(args.nc_path).expanduser().resolve()
    # out_dir = Path(args.out).expanduser().resolve()
    # out_dir.mkdir(parents=True, exist_ok=True)

    # if not nc_path.exists():
    #     raise FileNotFoundError(f"NetCDF not found: {nc_path}")

    # Open dataset
    nc_path = r'D:\NOAA\GitHub\sar_drift_output\output\SARIceDrift_EG125_2025350T0000_2025350T2359_gfilter1.nc'
    out_dir = r'D:\NOAA\GitHub\read_sar_drift_netcdf\output'
    ds = xr.open_dataset(nc_path)
    


    # 1) properties
    print_dataset_properties(ds)

    # 2) layers
    list_layers(ds)

    # 3) plot image and save PNG
    plot_netcdf(
        ds=ds,
        base_name = os.path.splitext(os.path.basename(nc_path))[0],
        output_dir=out_dir,
        step=6
    )
    # # 3-5) export + plot + png
    # print("\n=== EXPORT + PLOT ===")
    # total_png = 0
    # total_npy = 0

    # for varname in ds.data_vars:
    #     da = ds[varname]

    #     if da.ndim < 2:
    #         print(f"Skipping {varname} (ndim={da.ndim} < 2)")
    #         continue

    #     xy = choose_2d_dims(da)
    #     if xy is None:
    #         print(f"Skipping {varname} (cannot identify 2D dims)")
    #         continue

    #     print(f"\nVariable: {varname} | plot dims: {xy} | ndim={da.ndim} | shape={da.shape}")

    #     # slices = iter_2d_slices(da, xy_dims=xy, max_slices=args.max_slices)
    #     slices = iter_2d_slices(da, xy_dims=xy, max_slices=10)
    #     # print(f"  -> exporting/plotting {len(slices)} slice(s) (limit={args.max_slices})")

    #     for idx, sl in slices:
    #         arr = sl.values

    #         npy_path = export_array(out_dir, varname, idx, arr)
    #         total_npy += 1

    #         png_path = plot_and_save_png(out_dir, varname, idx, sl)
    #         total_png += 1

    #         print(f"    saved: {npy_path.name} | {png_path}")

    ds.close()
    print(f"\nDone.\nSaved arrays: {total_npy}  |  Saved PNGs: {total_png}")
    print(f"Output folder: {out_dir}")


if __name__ == "__main__":
    main()
