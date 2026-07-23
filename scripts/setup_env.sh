#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — probe this machine for MieWorkbench's pinned tool paths and
# write <repo>/miewb.env (the machine-path single-source-of-truth consumed by
# scripts/common.py). Interactive by default when run at a terminal; use
# --non-interactive for CI / scripted installs. See miewb.env.example for the
# file's full contract and scripts/miewb_env.sh to load it into a shell.
#
# Resolution order per key: explicit CLI flag > existing miewb.env value
# (idempotent re-run) > machine probe. In interactive mode you're shown the
# resolved candidate and can accept it, type a different path, or (for the
# optional keys) mark it deliberately absent with '-'.
# =============================================================================
set -euo pipefail

REPO="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# Output path. MIEWB_SETUP_OUT is an undocumented internal override used by
# this script's own test suite to avoid touching the real miewb.env — not
# part of the public contract, deliberately omitted from --help.
OUT="${MIEWB_SETUP_OUT:-$REPO/miewb.env}"

NON_INTERACTIVE=0
PRINT_ONLY=0

FREECAD_FLAG_SET=0;   FREECAD_FLAG_VAL=""
OPTICS_FLAG_SET=0;    OPTICS_FLAG_VAL=""
PVPYTHON_FLAG_SET=0;  PVPYTHON_FLAG_VAL=""
NVCC_FLAG_SET=0;      NVCC_FLAG_VAL=""
ARCH_FLAG_SET=0;      ARCH_FLAG_VAL=""

usage() {
    cat <<'EOF'
Usage: scripts/setup_env.sh [options]

Probe this machine for MieWorkbench's pinned external tools and write
<repo>/miewb.env. See miewb.env.example for the file contract.

Options:
  --freecad PATH        FreeCAD AppImage / executable.
                         (required — cannot be set to '')
  --optics-python PATH  optics-env python (numpy/scipy/torch-CUDA/h5py).
                         (required — cannot be set to '')
  --pvpython PATH       ParaView pvpython. '' = deliberately not installed.
  --nvcc PATH           CUDA nvcc (>=13). '' = deliberately not installed.
  --cuda-arch N         GPU SM arch, e.g. 89. '' = deliberately not installed.
  --non-interactive     Never prompt; a missing required tool exits 2 and
                         writes nothing.
  --print               Print the file that WOULD be written and exit;
                         writes nothing.
  -h, --help             Show this help and exit.

Precedence per key: this flag > the existing miewb.env value (so re-running
is idempotent) > a machine probe. Run with no flags at a terminal for an
interactive walkthrough.
EOF
}

need_arg() {
    # $1 = flag name being parsed, $2 = remaining arg count after it
    if [ "$2" -lt 1 ]; then
        echo "setup_env.sh: $1 requires an argument" >&2
        exit 2
    fi
}

while [ $# -gt 0 ]; do
    case "$1" in
        --freecad)
            need_arg "$1" "$(($#-1))"
            FREECAD_FLAG_SET=1; FREECAD_FLAG_VAL="$2"; shift 2 ;;
        --optics-python)
            need_arg "$1" "$(($#-1))"
            OPTICS_FLAG_SET=1; OPTICS_FLAG_VAL="$2"; shift 2 ;;
        --pvpython)
            need_arg "$1" "$(($#-1))"
            PVPYTHON_FLAG_SET=1; PVPYTHON_FLAG_VAL="$2"; shift 2 ;;
        --nvcc)
            need_arg "$1" "$(($#-1))"
            NVCC_FLAG_SET=1; NVCC_FLAG_VAL="$2"; shift 2 ;;
        --cuda-arch)
            need_arg "$1" "$(($#-1))"
            ARCH_FLAG_SET=1; ARCH_FLAG_VAL="$2"; shift 2 ;;
        --non-interactive)
            NON_INTERACTIVE=1; shift ;;
        --print)
            PRINT_ONLY=1; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "setup_env.sh: unrecognized argument: $1" >&2
            usage >&2
            exit 2 ;;
    esac
done

# Interactive only when stdin is a terminal and the caller didn't opt out.
INTERACTIVE=1
if [ "$NON_INTERACTIVE" = "1" ] || [ ! -t 0 ]; then
    INTERACTIVE=0
fi

# ---------------------------------------------------------------------------
# Parse an existing miewb.env-style file line by line (never `source` it —
# values must stay literal, exactly like scripts/common.py's load_env_file
# and scripts/miewb_env.sh see them).
# ---------------------------------------------------------------------------
declare -A EXIST

parse_existing() {
    local path="$1" line stripped key val
    EXIST=()
    [ -f "$path" ] || return 0
    while IFS= read -r line || [ -n "$line" ]; do
        stripped="$(printf '%s' "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
        [ -z "$stripped" ] && continue
        case "$stripped" in
            '#'*) continue ;;
        esac
        case "$stripped" in
            *=*) ;;
            *) continue ;;
        esac
        key="${stripped%%=*}"
        key="$(printf '%s' "$key" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
        case "$key" in
            MIEWB_*) ;;
            *) continue ;;
        esac
        case "$key" in
            *[!A-Z_]*) continue ;;
        esac
        val="${stripped#*=}"
        EXIST["$key"]="$val"
    done < "$path"
}

parse_existing "$OUT"

# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------
probe_freecad() {
    local c f
    for c in FreeCAD freecad; do
        if f="$(command -v "$c" 2>/dev/null)"; then
            printf '%s\n' "$f"
            return 0
        fi
    done
    shopt -s nullglob
    local -a cands=(
        /home3/freecad/*.AppImage
        "$HOME"/*.AppImage
        /opt/FreeCAD*/*.AppImage
        /opt/*.AppImage
        /home3/freecad/squashfs-root/AppRun
        "$HOME"/squashfs-root/AppRun
        /opt/FreeCAD*/squashfs-root/AppRun
        /opt/squashfs-root/AppRun
    )
    shopt -u nullglob
    for f in "${cands[@]}"; do
        if [ -x "$f" ]; then
            printf '%s\n' "$f"
            return 0
        fi
    done
    return 1
}

probe_optics_python() {
    local -a cands=("/home3/optics/env/bin/python")
    if [ -n "${CONDA_PREFIX:-}" ]; then
        cands+=("$CONDA_PREFIX/bin/python")
    fi
    local p
    for p in "${cands[@]}"; do
        if [ -x "$p" ] && "$p" -c "import numpy, scipy, h5py" >/dev/null 2>&1; then
            printf '%s\n' "$p"
            return 0
        fi
    done
    return 1
}

warn_optics_python_torch() {
    local p="$1"
    [ -n "$p" ] && [ -x "$p" ] || return 0
    if ! "$p" -c "import torch" >/dev/null 2>&1; then
        echo "NOTE: $p has no working 'import torch' (GPU-accelerated gather" \
             "will be unavailable) — continuing anyway." >&2
    fi
}

probe_pvpython() {
    local f
    if f="$(command -v pvpython 2>/dev/null)"; then
        printf '%s\n' "$f"
        return 0
    fi
    shopt -s nullglob
    local -a cands=(
        /home3/paraview/*/bin/pvpython
        /opt/ParaView*/bin/pvpython
        "$HOME"/ParaView*/bin/pvpython
    )
    shopt -u nullglob
    for f in "${cands[@]}"; do
        if [ -x "$f" ]; then
            printf '%s\n' "$f"
            return 0
        fi
    done
    return 1
}

probe_nvcc() {
    shopt -s nullglob
    local -a dirs=(/usr/local/cuda-*)
    shopt -u nullglob
    local d ver best_dir="" best_ver="" hi
    for d in "${dirs[@]}"; do
        [ -x "$d/bin/nvcc" ] || continue
        ver="${d##*/cuda-}"
        if [ -z "$best_dir" ]; then
            best_dir="$d"; best_ver="$ver"
            continue
        fi
        hi="$(printf '%s\n%s\n' "$best_ver" "$ver" | sort -V | tail -n1)"
        if [ "$hi" = "$ver" ] && [ "$ver" != "$best_ver" ]; then
            best_dir="$d"; best_ver="$ver"
        fi
    done
    if [ -n "$best_dir" ]; then
        printf '%s\n' "$best_dir/bin/nvcc"
        return 0
    fi
    if [ -x /usr/bin/nvcc ]; then
        echo "NOTE: only /usr/bin/nvcc found — this is likely the distro" \
             "CUDA 11.x nvcc, not the required >=13 toolkit; not" \
             "auto-selected. Pass --nvcc explicitly if you really want it." >&2
    fi
    return 1
}

probe_cuda_arch() {
    command -v nvidia-smi >/dev/null 2>&1 || return 1
    local raw
    raw="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null)" || return 1
    [ -n "$raw" ] || return 1
    local -a caps=()
    local line cleaned
    while IFS= read -r line; do
        cleaned="$(printf '%s' "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
        [ -n "$cleaned" ] && caps+=("$cleaned")
    done <<< "$raw"
    [ "${#caps[@]}" -gt 0 ] || return 1
    local first="${caps[0]}" c mismatch=0
    for c in "${caps[@]}"; do
        [ "$c" = "$first" ] || mismatch=1
    done
    if [ "$mismatch" = "1" ]; then
        echo "NOTE: GPUs report different compute capabilities:" \
             "${caps[*]} — using the first (${first})." >&2
    fi
    printf '%s\n' "${first//./}"
    return 0
}

# ---------------------------------------------------------------------------
# Interactive prompt helper
# ---------------------------------------------------------------------------
prompt_for_key() {
    # $1 = display label, $2 = candidate (may be empty), $3 = allow_absent (0/1),
    # $4 = is_path (0/1, default 1) -- non-path values (e.g. a bare SM
    # architecture number like "89") must skip the on-disk existence check.
    local label="$1" candidate="$2" allow_absent="$3" is_path="${4:-1}"
    local reply value confirm
    while true; do
        if [ -n "$candidate" ]; then
            printf '%s\n  candidate: %s\n' "$label" "$candidate" >&2
        else
            printf '%s\n  candidate: (none found)\n' "$label" >&2
        fi
        if [ "$allow_absent" = "1" ]; then
            printf "  [Enter=accept / type a path / '-' = not installed]: " >&2
        else
            printf "  [Enter=accept / type a path] (required, cannot be absent): " >&2
        fi
        IFS= read -r reply || reply=""
        if [ -z "$reply" ]; then
            value="$candidate"
        elif [ "$reply" = "-" ]; then
            if [ "$allow_absent" != "1" ]; then
                echo "  This tool is required and cannot be configured absent." >&2
                continue
            fi
            value=""
        else
            value="$reply"
        fi
        if [ "$allow_absent" != "1" ] && [ -z "$value" ]; then
            echo "  This tool is required — please provide a path." >&2
            continue
        fi
        if [ "$is_path" = "1" ] && [ -n "$value" ] && [ ! -e "$value" ]; then
            printf "  %s does not exist on disk — use it anyway? [y/N]: " "$value" >&2
            IFS= read -r confirm || confirm=""
            case "$confirm" in
                y|Y|yes|YES) ;;
                *) echo "  Not accepted; try again." >&2; continue ;;
            esac
        fi
        printf '%s' "$value"
        return 0
    done
}

# ---------------------------------------------------------------------------
# Resolve each key: flag > existing > probe (+ interactive confirmation).
#
# NOTE: these are invoked as VALUE="$(resolve_required ...)" — command
# substitution runs in a subshell, so a resolver must never be the sole
# place that records a required-key failure (array writes there would be
# lost). It only ever prints its result (possibly empty); the caller in
# the main shell decides whether an empty required result is fatal.
# ---------------------------------------------------------------------------
MISSING=()

resolve_required() {
    # $1=key $2=label $3=flag_set $4=flag_val $5=probe_fn -> prints value
    local key="$1" label="$2" flag_set="$3" flag_val="$4" probe_fn="$5"
    local value="" have=0 candidate=""

    if [ "$flag_set" = "1" ]; then
        value="$flag_val"; have=1
    elif [ -n "${EXIST[$key]+x}" ]; then
        value="${EXIST[$key]}"; have=1
    fi

    if [ "$flag_set" != "1" ] && [ "$INTERACTIVE" = "1" ]; then
        if [ "$have" = "1" ] && [ -n "$value" ]; then
            candidate="$value"
        else
            candidate="$("$probe_fn" 2>/dev/null || true)"
        fi
        value="$(prompt_for_key "$label" "$candidate" 0)"
        have=1
        [ -n "$value" ] || have=0
    fi

    if [ "$have" != "1" ] || [ -z "$value" ]; then
        if [ "$flag_set" != "1" ] && [ "$INTERACTIVE" != "1" ]; then
            candidate="$("$probe_fn" 2>/dev/null || true)"
            if [ -n "$candidate" ]; then
                printf '%s\n' "$candidate"
                return 0
            fi
        fi
        printf ''
        return 0
    fi
    printf '%s\n' "$value"
}

resolve_optional() {
    # $1=key $2=label $3=flag_set $4=flag_val $5=probe_fn $6=is_path (0/1,
    # default 1 -- pass 0 for non-path values like the bare CUDA arch
    # number) -> prints value (possibly empty)
    local key="$1" label="$2" flag_set="$3" flag_val="$4" probe_fn="$5"
    local is_path="${6:-1}"
    local value="" have=0 candidate=""

    if [ "$flag_set" = "1" ]; then
        printf '%s\n' "$flag_val"
        return 0
    fi
    if [ -n "${EXIST[$key]+x}" ]; then
        value="${EXIST[$key]}"; have=1
    fi

    if [ "$INTERACTIVE" = "1" ]; then
        if [ "$have" = "1" ]; then
            candidate="$value"
        else
            candidate="$("$probe_fn" 2>/dev/null || true)"
        fi
        prompt_for_key "$label" "$candidate" 1 "$is_path"
        return 0
    fi

    if [ "$have" = "1" ]; then
        printf '%s\n' "$value"
        return 0
    fi
    candidate="$("$probe_fn" 2>/dev/null || true)"
    if [ -n "$candidate" ]; then
        printf '%s\n' "$candidate"
    else
        echo "NOTE: $label not found on this machine; configuring absent" \
             "(empty). Pass a flag or edit miewb.env to set it later." >&2
        printf ''
    fi
}

FREECAD_VAL="$(resolve_required MIEWB_FREECAD "FreeCAD AppImage/executable" \
    "$FREECAD_FLAG_SET" "$FREECAD_FLAG_VAL" probe_freecad)"
[ -n "$FREECAD_VAL" ] || MISSING+=("MIEWB_FREECAD")
OPTICS_VAL="$(resolve_required MIEWB_OPTICS_PYTHON "optics-env python (numpy/scipy/h5py)" \
    "$OPTICS_FLAG_SET" "$OPTICS_FLAG_VAL" probe_optics_python)"
[ -n "$OPTICS_VAL" ] || MISSING+=("MIEWB_OPTICS_PYTHON")
if [ -n "$OPTICS_VAL" ]; then
    warn_optics_python_torch "$OPTICS_VAL"
fi
PVPYTHON_VAL="$(resolve_optional MIEWB_PVPYTHON "ParaView pvpython" \
    "$PVPYTHON_FLAG_SET" "$PVPYTHON_FLAG_VAL" probe_pvpython)"
NVCC_VAL="$(resolve_optional MIEWB_NVCC "CUDA nvcc (>=13)" \
    "$NVCC_FLAG_SET" "$NVCC_FLAG_VAL" probe_nvcc)"
ARCH_VAL="$(resolve_optional MIEWB_CUDA_ARCH "GPU SM architecture (nvidia-smi compute_cap)" \
    "$ARCH_FLAG_SET" "$ARCH_FLAG_VAL" probe_cuda_arch 0)"

if [ "${#MISSING[@]}" -gt 0 ]; then
    echo "setup_env.sh: could not resolve required tool path(s): ${MISSING[*]}" >&2
    echo "Pass --freecad / --optics-python explicitly, or re-run" \
         "interactively (drop --non-interactive)." >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Preserve any existing optional-override lines (uncommented) verbatim;
# everything else in that block stays commented, matching miewb.env.example.
# ---------------------------------------------------------------------------
optional_line() {
    # $1=key $2=comment-only placeholder line
    local key="$1" placeholder="$2"
    if [ -n "${EXIST[$key]+x}" ]; then
        printf '%s=%s\n' "$key" "${EXIST[$key]}"
    else
        printf '%s\n' "$placeholder"
    fi
}

GUI_PYTHON_LINE="$(optional_line MIEWB_GUI_PYTHON '# MIEWB_GUI_PYTHON=')"
GEOMETRY_DIR_LINE="$(optional_line MIEWB_GEOMETRY_DIR '# MIEWB_GEOMETRY_DIR=')"
RESULTS_DIR_LINE="$(optional_line MIEWB_RESULTS_DIR '# MIEWB_RESULTS_DIR=')"
OPTPROPS_DIR_LINE="$(optional_line MIEWB_OPTPROPS_DIR '# MIEWB_OPTPROPS_DIR=')"
CENGINE_LINE="$(optional_line MIEWB_CENGINE '# MIEWB_CENGINE=')"

render_env_file() {
    cat <<EOF
# =============================================================================
# miewb.env — MieWorkbench machine paths for THIS machine.
#
# Generated by scripts/setup_env.sh (re-run any time to refresh). Gitignored:
# it describes ONE machine, never the project — see miewb.env.example for
# the template/full contract. The GUI's Settings dialog also edits this file
# in place (comment-preserving), so hand edits are safe too.
#
# Precedence: an exported MIEWB_* environment variable beats a line here.
# Format rules: KEY=value, one per line; values are LITERAL — no quotes, no
# \$variable interpolation, no shell syntax; use absolute paths. An EMPTY
# value (KEY=) means "this tool is deliberately not installed here".
# =============================================================================

# --- required external tools -------------------------------------------------
# FreeCAD 1.1.1 AppImage (runs extract_geometry / permute_model / fcserver).
MIEWB_FREECAD=$FREECAD_VAL

# The optics environment's python (numpy/scipy/torch-CUDA/miepython/h5py;
# runs run_trace / post_process / the engine test suite). See INSTALL.md §3.2.
MIEWB_OPTICS_PYTHON=$OPTICS_VAL

# ParaView's pvpython (viz stage renders). Leave EMPTY if this machine has
# no ParaView — the pipeline skips viz cleanly.
MIEWB_PVPYTHON=$PVPYTHON_VAL

# --- CUDA (optional — leave both empty on CPU-only machines) -----------------
# nvcc from a CUDA >= 13 toolkit (NOT the distro /usr/bin/nvcc if it's 11.x).
MIEWB_NVCC=$NVCC_VAL
# GPU SM architecture for the C engine, e.g. 89 for RTX 4090 (compute 8.9);
# setup_env.sh reads it from nvidia-smi.
MIEWB_CUDA_ARCH=$ARCH_VAL

# --- optional overrides (defaults derive from the repo location) -------------
# The GUI virtualenv python (default: <repo>/env/bin/python).
$GUI_PYTHON_LINE
# Geometry cache / results / optical-properties library directories
# (defaults: <repo>/geometry, <repo>/results, <repo>/opticalproperties).
$GEOMETRY_DIR_LINE
$RESULTS_DIR_LINE
$OPTPROPS_DIR_LINE
# C-engine binary override (default: <repo>/cengine/build/miewb-trace).
$CENGINE_LINE
EOF
}

if [ "$PRINT_ONLY" = "1" ]; then
    render_env_file
    exit 0
fi

TMP="$(mktemp "$(dirname -- "$OUT")/.miewb.env.XXXXXX")"
trap 'rm -f "$TMP"' EXIT
render_env_file > "$TMP"
mv -f "$TMP" "$OUT"
trap - EXIT

echo "Wrote $OUT" >&2

if ! MIEWB_ENV_FILE="$OUT" python3 "$REPO/scripts/common.py"; then
    echo "setup_env.sh: validation of $OUT via scripts/common.py FAILED" >&2
    exit 1
fi

echo "" >&2
echo "Load these paths into your shell with:  source scripts/miewb_env.sh" >&2
