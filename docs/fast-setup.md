# Fast OCaml Development Setup

FIXME: NOTE: **This functionality is not available yet, ignore this document.**

This guide explains how to use pre-built relocatable opam switches to dramatically speed up OCaml environment setup for CI, cloud development, and local installation.

## The Problem

A typical `opam install . --deps-only` for a project like OCANNL takes **10-20 minutes** because:

1. The OCaml compiler must be built from source
2. All dependencies must be compiled
3. This happens on every fresh CI run or cloud session

## The Solution: Relocatable OCaml

As of late 2025, OCaml supports **relocatable switches** — pre-built compiler and package installations that can be moved between machines and paths. This means:

- Build the switch once → reuse everywhere
- CI setup: **~15 min → ~1-2 min**
- Claude Code cloud: Ready to code instead of waiting

## Quick Start

### For Users

To install OCANNL with a pre-built switch:

```bash
# Install opam
bash -c "sh <(curl -fsSL https://opam.ocaml.org/install.sh)"
opam init -y --bare

# Download pre-built switch (adjust version as needed)
RELEASE_URL="https://github.com/ahrefs/ocannl/releases/download/switch-cache-v1"
curl -fSL "$RELEASE_URL/ocannl-switch-5.4.0-linux-x86_64-slim.tar.zst" \
  | tar -I zstd -xf - -C ~/.opam

# Link to your project
cd your-project
opam switch link ocannl-switch .
eval $(opam env)

# Ready to build!
dune build
```

### Available Platforms

| Platform | Tarball |
|----------|---------|
| Linux x86_64 | `ocannl-switch-5.4.0-linux-x86_64-slim.tar.zst` |
| macOS ARM64 | `ocannl-switch-5.4.0-macos-arm64-slim.tar.zst` |

The `-slim` variants exclude build artifacts and documentation for faster downloads.

## For CI (GitHub Actions)

Replace your slow `setup-ocaml` step with:

```yaml
- name: Install opam
  run: |
    bash -c "sh <(curl -fsSL https://opam.ocaml.org/install.sh)"
    opam init -y --bare --disable-sandboxing

- name: Restore cached switch
  run: |
    curl -fSL "https://github.com/ahrefs/ocannl/releases/download/switch-cache-v1/ocannl-switch-5.4.0-linux-x86_64-slim.tar.zst" \
      | tar -I zstd -xf - -C ~/.opam
    opam switch link ocannl-switch .

- name: Build
  run: |
    eval $(opam env)
    dune build
```

See `.github/workflows/ci-fast.yml` for a complete example.

## For Claude Code Cloud

The repository includes a SessionStart hook that automatically sets up the OCaml environment. When you start a Claude Code cloud session:

1. The hook downloads the pre-built switch
2. Environment variables are persisted for subsequent commands
3. You're ready to code in ~1-2 minutes

Configuration is in `.claude/settings.json` and `scripts/setup-ocaml-env.sh`.

## Building Your Own Switch Cache

### Prerequisites

- OCaml 5.4+ with [dra27's relocatable backport](https://github.com/dra27/opam-repository/tree/relocatable), OR
- OCaml 5.5+ which has relocatability built-in

### Manual Build

```bash
# Initialize opam with relocatable repository (for OCaml < 5.5)
opam init -y --bare \
  --repos=relocatable=git+https://github.com/dra27/opam-repository.git#relocatable,default

# Create switch
opam switch create ocannl-switch ocaml.5.4.0 -y

# Install your dependencies
eval $(opam env --switch=ocannl-switch)
opam install . -y --deps-only --with-test --with-doc

# Package as tarball
tar -I 'zstd -19 -T0' -cf ocannl-switch-5.4.0-linux-x86_64.tar.zst \
  -C ~/.opam ocannl-switch
```

### Automated Build with GitHub Actions

Use `.github/workflows/build-switch-cache.yml`:

1. Go to Actions → Build Switch Cache
2. Click "Run workflow"
3. Fill in OCaml version and release tag
4. Wait for the workflow to complete
5. Find tarballs in the new GitHub Release

## How It Works

### Relocatable OCaml (Technical Details)

Traditional OCaml installations embed absolute paths at compile time:
- `ocamlopt -where` returns a hardcoded path
- Moving the installation breaks everything

Relocatable OCaml ([RFC](https://github.com/ocaml/RFCs/pull/53), [implementation](https://github.com/ocaml/ocaml/pull/14243)) solves this by:
- Computing paths relative to the executable at runtime
- Using `--with-relative-libdir` and `--enable-runtime-search` configure flags
- Allowing switches to be moved, renamed, or distributed as tarballs

### What's Included in the Tarball

The full tarball contains:
- OCaml compiler and runtime
- All opam-installed packages
- Dev tools (LSP, formatter, merlin)
- Build artifacts and documentation

The slim tarball excludes:
- `.opam-switch/build/` (intermediate build files)
- `doc/` (generated documentation)
- `.cmt`/`.cmti` files (typed trees for editors)

### Compatibility Notes

- **Path independence**: The switch works regardless of where it's extracted
- **Same architecture required**: A Linux x86_64 tarball won't work on ARM
- **opam version**: Built with opam 2.2.x, should work with 2.1.x+
- **System dependencies**: Some packages may need system libraries (e.g., libffi, gmp)

## Updating the Cache

When dependencies change:

1. Update your opam files
2. Re-run the `build-switch-cache.yml` workflow with a new tag
3. Update `SWITCH_CACHE_TAG` in `ci-fast.yml` and `setup-ocaml-env.sh`

## Troubleshooting

### "Switch not found" after extraction

Make sure opam is initialized:
```bash
opam init -y --bare
```

### "Permission denied" on scripts

```bash
chmod +x scripts/setup-ocaml-env.sh
```

### Different home directory path

The relocatable switch doesn't care about the exact path. If you extracted to `/home/runner/.opam` but your home is `/home/claude`, just:
```bash
mv ~/.opam/ocannl-switch ~/.opam/ocannl-switch-old
tar -I zstd -xf tarball.tar.zst -C ~/.opam
```

### Missing system dependencies

Some packages need system libraries. On Ubuntu:
```bash
sudo apt-get install libffi-dev libgmp-dev pkg-config
```

## References

- [Relocatable OCaml announcement](https://discuss.ocaml.org/t/relocatable-ocaml/17253)
- [RFC for relocatable compiler](https://github.com/ocaml/RFCs/pull/53)
- [dra27's relocatable backport](https://github.com/dra27/opam-repository/tree/relocatable)
- [Discussion on CI performance](https://discuss.ocaml.org/t/is-it-normal-for-ocaml-setup-ocaml-v3-to-take-over-8-minutes/17419)
- [Claude Code cloud documentation](https://code.claude.com/docs/en/claude-code-on-the-web)
