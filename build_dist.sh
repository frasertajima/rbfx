#!/usr/bin/env bash
# Rebuilds this folder's binary-only distribution from rbfx-py's source.
# Run this after any source change; it's how test/ stays in sync — nothing
# here is hand-edited.
#
# What it produces (no .rs source, no Cargo.toml, in either):
#   test/rbfx-<version>-<tag>.whl   — `pip install` this
#   test/rbfx_binary/               — drop-in: sys.path.append() + `import rbfx`, no pip needed
#
# Both bundle libgp_solver.so and use an $ORIGIN-relative rpath (see
# rbfx-py/build.rs's RBFX_DIST_BUILD mode) so they're self-contained: no
# dependency on this dev machine's rbfx/.native symlink or gp_engine's exact
# path, as long as the target machine has the same CUDA/NVIDIA HPC SDK
# runtime installed (libcusolver/libcublas/libcudart/nvfortran runtime —
# those are NOT bundled, matching how e.g. cupy wheels don't bundle CUDA
# itself either).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RBFX_PY="$HERE/../rbfx-py"

echo "== building portable wheel (RBFX_DIST_BUILD=1, auditwheel skip) =="
( cd "$RBFX_PY" && cargo clean -p rbfx-py >/dev/null 2>&1 || true
  RBFX_DIST_BUILD=1 maturin build --release --auditwheel skip )

WHEEL="$(ls -t "$RBFX_PY"/target/wheels/rbfx-*.whl | head -1)"
rm -f "$HERE"/rbfx-*.whl
cp "$WHEEL" "$HERE/"
echo "copied $(basename "$WHEEL") -> test/"

echo "== extracting the unpacked (no-pip) flavor =="
rm -rf "$HERE/rbfx_binary"
mkdir -p "$HERE/rbfx_binary"
tmp="$(mktemp -d)"
( cd "$tmp" && unzip -q "$WHEEL" )
cp -r "$tmp/rbfx" "$HERE/rbfx_binary/rbfx"
rm -rf "$tmp"
echo "wrote test/rbfx_binary/rbfx/ (drop-in: sys.path + import rbfx, no pip)"

echo "== restoring the normal dev build (RBFX_DIST_BUILD unset) =="
( cd "$RBFX_PY" && cargo clean -p rbfx-py >/dev/null 2>&1 && maturin develop --release )

echo "done."
