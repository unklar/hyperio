# Window Aggregation Mode for HSI Index Computation

## Summary

Add a `window_agg` parameter (default `"mean"`) to `compute_index` and related functions, allowing users to choose how window band data is aggregated: mean, median, or sum.

## Motivation

Currently, when `window > 0`, the `average_window` utility always computes the mean of neighboring bands. For some spectral indices, median (robust to outliers) or sum (preserving total signal) may be more appropriate.

## Design

### New parameter: `window_agg`

- Type: `str`
- Default: `"mean"`
- Valid values: `("mean", "median", "sum")`
- Behavior:
  - `"mean"` — `.mean(axis=2)` (current behavior)
  - `"median"` — `np.median(..., axis=2)`
  - `"sum"` — `.sum(axis=2)`
- Raises `ValueError` for invalid values

### Affected functions

1. **`_utils.average_window(cube, band_idx, window, agg="mean")`** — add `agg` param, dispatch aggregation
2. **`HSI.nearest_band(wavelength, window=0, window_agg="mean")`** — forward to `average_window`
3. **`HSI.bands_from_wavelengths(wavelengths, window=0, window_agg="mean")`** — forward to `nearest_band`
4. **`HSI.compute_index(formula, **kwargs)`** — pop `window_agg` from kwargs, forward through the chain
5. **`HSI.rgb(wl_red, ..., window=0, window_agg="mean")`** — forward to `bands_from_wavelengths`
6. Predefined index methods (ndvi, ndwi, evi, etc.) — no changes needed (forward `**kw` already)

### Backward compatibility

Fully preserved. Default `"mean"` matches current behavior.

## Testing

- Unit test for `average_window` with all three aggregation modes
- Unit test for `compute_index` with `window_agg="median"` and `window_agg="sum"`
- Test that invalid `window_agg` raises `ValueError`
