# rbfx — binary-only distribution

Two ways to get `rbfx` onto another machine **without shipping the Rust
source** (no `.rs` files, no `Cargo.toml` in either flavor below). Matches the
`gp_engine/test`/`MPDOK/test` "compiled binary + wrapper + README" pattern.

Both flavors bundle `libgp_solver.so` (a copy of `gp_engine/gp_solver.so`) and
use an `$ORIGIN`-relative runtime path — they find that copy next to
themselves, not via this dev machine's `rbfx/.native` symlink or a fixed
`gp_engine` path. Regenerate both from source with `./build_dist.sh`.

## Option 1 — `pip install`

```bash
pip install rbfx-0.1.0-cp314-cp314-linux_x86_64.whl
python3 -c "import rbfx; print(rbfx.Rbfx)"
```

Tag in the filename (`cp314-cp314-linux_x86_64`) must match the target's
Python version/platform — rebuild with `./build_dist.sh` under the target
Python if it doesn't.

## Option 2 — drop-in, no pip

```bash
cp -r rbfx_binary/rbfx /path/in/your/project/
python3 -c "
import sys; sys.path.insert(0, '/path/in/your/project')
import rbfx
"
```

`rbfx_binary/rbfx/` is just the wheel's payload unpacked — copy the whole
`rbfx/` folder anywhere on `sys.path`.

## What's NOT bundled

The CUDA/NVIDIA HPC SDK runtime (`libcudart`, `libcublas`, `libcusolver`, the
nvfortran runtime libs) — these must already be installed on the target
machine, same as this dev machine's toolchain (this mirrors how e.g. `cupy`
wheels don't bundle the CUDA toolkit itself). This distribution is for
**other machines you control** with that toolchain already present — not a
zero-dependency public release.

## Rebuilding

```bash
./build_dist.sh
```

Builds `rbfx-py` with `RBFX_DIST_BUILD=1` (switches the runtime path from the
dev-machine-specific `.native` symlink to `$ORIGIN`) and `--auditwheel skip`
(prevents maturin from trying to vendor the entire CUDA/HPC SDK toolchain
into the wheel — we want the target machine's own install, not a copy of
it), then restores the normal dev build (`maturin develop --release`)
afterward so this doesn't disturb your local dev environment.
