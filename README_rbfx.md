# rbfx — dense RBF/kriging interpolation, Rust + GPU, callable from Jupyter

A compact Rust library wrapping MPDOK/gp_engine's proven mixed-precision dense
solver (`gp_solver.so`: FP32 Cholesky factor + FP64 iterative refinement,
CUDA Fortran/cuBLAS/cuSOLVER) so it's usable as a generic dense
scattered-data-interpolation kernel — from a plain Rust program, or from
Python/Jupyter via a PyO3 wheel — rather than only through this codebase's
existing app-specific ctypes wrappers (`gp_engine`, `MPDOK/kriging`,
`rbf_pointcloud`, `rbf_spatial`).

No new numerics were written: this crate is an FFI/packaging layer over the
same `gp_solver.so` those Python projects already call, validated against
that exact library via a parity test (see below).

## What's in this folder

| Path | What |
|---|---|
| `rbfx-core/` | Plain Rust crate — FFI over `gp_solver.so` + a port of `gp_core.py`'s iterative-refinement loop. Usable standalone (`cargo add --path`), no Python required. |
| `rbfx-py/` | PyO3 bindings (`cdylib`) — `import rbfx` from Python/Jupyter. Build with `maturin develop --release`. |
| `bench/parity_test.py` | Validates `rbfx` against `gp_fortran.py` (the existing CUDA-Fortran-backed Python oracle) — same `.so`, so `alpha`/`relres`/`logdet` must agree to float precision. |
| `bench/benchmark.py` | N-sweep: wall-clock time **and** accuracy (RMSE vs the true field) vs `scipy.interpolate.RBFInterpolator`. Existing MPDOK benchmark scripts only compared time — this adds the accuracy axis. |
| `rbfx_demo.ipynb` | Quickstart notebook (fit/predict/error-handling/benchmark summary). |
| `test/` | **Binary-only distribution** — a wheel (`pip install`) and an unpacked drop-in flavor, no `.rs` source in either. See `test/README.md`. |

## Quickstart

```python
import numpy as np
import rbfx

X = ...  # (n, d) point coordinates
y = ...  # (n,) observed values

fit = rbfx.Rbfx(X, y, kernel="matern32", ell=2.0, sigma_f=1.0, sigma_n2=1e-4,
                tol=1e-11, max_ir=10)
print(fit.relres, fit.n_ir, fit.converged, fit.logdet)

pred = fit.predict(Xq)  # (m,) predictive mean at query points Xq (m, d)
```

`kernel` is one of `"rbf"`, `"matern32"`, `"matern52"` (gp_core.py's
`_KINDS`), isotropic lengthscale `ell` (ARD not exposed in this v1 API).

## Build

```bash
# 1. rbfx-core needs gp_solver.so already built (gp_engine/Makefile).
# 2. Build + install the Python wheel:
cd rbfx-py
pip install maturin
maturin develop --release
```

`rbfx-core/build.rs` locates `gp_solver.so` via `RBFX_GP_SOLVER_DIR` (default
`~/machine_learning/gp_engine`), and materializes a stable `libgp_solver.so`
symlink in `rbfx/.native/` so the normal `-L`/`-l` linker instructions work
against a `.so` that doesn't follow the `lib*.so` naming convention.

## Measured (this session, RTX-class GPU, 2026-07-18)

Gaussian kernel, `ell=2.0`, matched against `scipy.interpolate.RBFInterpolator`
(`kernel="gaussian"`, `epsilon` set to match `ell`) — full sweep in
`bench/results.json` / `bench/results_time.png` / `bench/results_accuracy.png`:

| N | rbfx t_total (s) | scipy t_total (s) | rbfx RMSE | scipy RMSE |
|---|---|---|---|---|
| 500 | 0.231 | 0.006 | 0.01767 | 0.01767 |
| 2,000 | 0.072 | 0.180 | 0.00559 | 0.00558 |
| 4,000 | — | 1.018 | — | 0.00319 |
| 8,000 | — | 6.782 | — | 0.00231 |
| 10,000 | 0.961 | — | 0.00213 | — |
| 20,000 | 2.226 | not run | 0.00146 | — |

Accuracy tracks scipy closely at matched N (both fit the same Gaussian-kernel
model) — the win is wall-clock time at scale, growing as scipy's dense CPU
solve (`O(N^3)`) becomes the bottleneck; small N favors scipy (CUDA context +
kernel-launch overhead dominates below a few thousand points).

## Limits

- **GPU required.** This crate only wraps `gp_solver.so` (no CPU fallback,
  by design — see the project plan) — there is no code path that runs
  without a CUDA-capable device.
- **d ≤ 32** (gp_solver.cuf's `MAX_D`, register-resident kernel).
- **Isotropic lengthscale only** in this v1 API — gp_core.py's ARD
  (per-dimension `ell`) isn't exposed yet.
- **Mean-only `predict`** — no predictive variance (would need
  `py_rbf_cross_build_f32` + a triangular solve; deferred until a benchmark
  needs it).
- **FP32 "kappa wall"**: ill-conditioned kernels (large N, large `ell`
  relative to point spacing) can fail to converge within `max_ir` — this
  is the same documented envelope as `gp_engine`/`MPDOK`, not a bug in this
  wrapper (see the parity test's `n=2000, kernel="rbf"` case, where both
  `rbfx` and the oracle agree on non-convergence to float precision).
- Single-GPU only (no multi-device dispatch).
