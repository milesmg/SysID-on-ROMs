#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JULIA_ROOT="$REPO_ROOT/Julia"
JULIA_VERSION="${JULIA_VERSION:-1.12.6}"
JULIA_TARBALL="julia-$JULIA_VERSION-linux-x86_64.tar.gz"
JULIA_URL="${JULIA_URL:-https://julialang-s3.julialang.org/bin/linux/x64/1.12/$JULIA_TARBALL}"
REPO_JULIA="$JULIA_ROOT/julia-$JULIA_VERSION/bin/julia"
PROJECT_DIR="$JULIA_ROOT/HPC_compatibility"
DEPOT_DIR="$JULIA_ROOT/depot"

mkdir -p "$DEPOT_DIR" "$DEPOT_DIR/dev"

if [ ! -x "$REPO_JULIA" ]; then
    cd "$JULIA_ROOT"
    if [ ! -f "$JULIA_TARBALL" ]; then
        if command -v curl >/dev/null 2>&1; then
            curl -L -o "$JULIA_TARBALL" "$JULIA_URL"
        else
            wget -O "$JULIA_TARBALL" "$JULIA_URL"
        fi
    fi
    tar -xzf "$JULIA_TARBALL"
fi
JULIA_EXE="$REPO_JULIA"

if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
    echo "Downloaded/extracted Linux Julia at $JULIA_ROOT/julia-$JULIA_VERSION"
    echo "Run this script on hpc1 to instantiate and precompile Julia/depot."
    exit 0
fi

export JULIA_DEPOT_PATH="$DEPOT_DIR:"
export JULIA_PKG_DEVDIR="$DEPOT_DIR/dev"
export JULIA_PKG_USE_CLI_GIT=true

echo "JULIA_EXE=$JULIA_EXE"
echo "JULIA_DEPOT_PATH=$JULIA_DEPOT_PATH"
"$JULIA_EXE" --version
"$JULIA_EXE" --project="$PROJECT_DIR" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
