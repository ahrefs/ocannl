#!/bin/bash
# Setup OCaml environment for Claude Code cloud sessions
#
# This script:
# 1. Installs opam if not present
# 2. Downloads and restores pre-built switch from GitHub Releases
# 3. Sets up environment variables for subsequent commands
#
# Expected runtime: ~1-2 minutes (vs ~15 min building from scratch)

set -e

# FIXME: Not ready yet
exit 0

# Only run in Claude Code cloud environment
if [ "$CLAUDE_CODE_REMOTE" != "true" ]; then
    echo "Not in Claude Code cloud environment, skipping OCaml setup"
    exit 0
fi

echo "=== Setting up OCaml environment for Claude Code cloud ==="

# Configuration - update these when you rebuild the switch
GITHUB_REPO="ahrefs/ocannl"
SWITCH_CACHE_TAG="switch-cache-v1"
OCAML_VERSION="5.4.0"
TARBALL_NAME="ocannl-switch-${OCAML_VERSION}-linux-x86_64-slim.tar.zst"

# Check if switch already exists
if opam switch list 2>/dev/null | grep -q "ocannl-switch"; then
    echo "Switch already exists, skipping download"
    opam switch link ocannl-switch . 2>/dev/null || true
else
    # Install opam if not present
    if ! command -v opam &> /dev/null; then
        echo "Installing opam..."
        bash -c "sh <(curl -fsSL https://opam.ocaml.org/install.sh)" -- --version 2.2.1
    fi

    # Initialize opam if needed
    if [ ! -d ~/.opam ]; then
        echo "Initializing opam..."
        opam init -y --bare --disable-sandboxing
    fi

    # Download and extract pre-built switch
    TARBALL_URL="https://github.com/${GITHUB_REPO}/releases/download/${SWITCH_CACHE_TAG}/${TARBALL_NAME}"
    echo "Downloading pre-built switch from: $TARBALL_URL"
    
    # Install zstd if not present
    if ! command -v zstd &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y zstd
    fi
    
    # Download and extract
    curl -fSL "$TARBALL_URL" | tar -I zstd -xf - -C ~/.opam
    
    # Link switch to project directory
    opam switch link ocannl-switch .
fi

# Verify installation
echo "Verifying OCaml installation..."
eval $(opam env)
ocamlopt -version
dune --version

# Persist environment variables for subsequent Claude bash commands
# This uses Claude Code's CLAUDE_ENV_FILE mechanism
if [ -n "$CLAUDE_ENV_FILE" ]; then
    echo "Persisting opam environment to CLAUDE_ENV_FILE..."
    opam env --shell=sh >> "$CLAUDE_ENV_FILE"
fi

echo "=== OCaml environment ready ==="
