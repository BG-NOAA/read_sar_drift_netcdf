# NetCDF Walkthrough (Jupyter Notebook)

This repo contains a hands-on Jupyter notebook that demonstrates how to open, inspect, analyze, visualize, and export data from a NetCDF (`.nc`) file using Python.

## What the notebook covers

The notebook walks through a typical NetCDF workflow:

1. **Install/import packages**
2. **Open a NetCDF file** with `xarray`
3. **Explore dataset structure**
   - dimensions and coordinates
   - variables and attributes
   - quick high-level stats
4. **Inspect variables in detail**
   - missing data (NaNs)
   - min/max/mean/median/std
   - slicing and selecting by time
5. **Plot data**
   - basic 2D plotting with Matplotlib
   - vector plotting (quiver) with Cartopy for map context (when coordinates are projected)
6. **Export for other Python tools**
   - export variable arrays to NumPy (`.npz`) for easy reuse

## Files

- `read_netcdf.ipynb` — main tutorial notebook
- `read_sar_drift_netcdf.ipynb` — script that will generate NumPy arrays and PNG from sar drift NetCDF files
- `requirements.txt` — Python dependencies for pip installation

## Quick start

### 1) Create and activate an environment (recommended)

**Conda**
```bash
conda create -n read-netcdf python=3.11 -y
conda activate read-netcdf

# Install dependencies from requirements.txt
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
