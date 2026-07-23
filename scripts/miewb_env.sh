# miewb_env.sh — source this (bash or sh) to load MieWorkbench machine
# paths into your shell:   source scripts/miewb_env.sh
#
# Exports MIEWB_INST_DIR (the repo root) and every MIEWB_* key from
# <repo>/miewb.env that is not already set in the environment (same
# env-wins precedence as scripts/common.py). The file is read line by
# line, NEVER shell-sourced: values stay literal, exactly as the Python
# parser sees them. Safe to source repeatedly; never exits your shell.
# Set MIEWB_ENV_QUIET=1 to silence the missing-file hint.

MIEWB_INST_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE:-$0}")/.." 2>/dev/null && pwd)"
# Plain sh (dash) has no BASH_SOURCE and $0 is the shell itself — validate
# the derived root and fall back to $PWD (sourcing from the repo root or a
# subdir is the documented usage).
if [ ! -f "$MIEWB_INST_DIR/scripts/miewb_env.sh" ]; then
    if [ -f "$PWD/scripts/miewb_env.sh" ]; then
        MIEWB_INST_DIR="$PWD"
    elif [ -f "$PWD/../scripts/miewb_env.sh" ]; then
        MIEWB_INST_DIR="$(CDPATH= cd -- "$PWD/.." && pwd)"
    fi
fi
export MIEWB_INST_DIR

if [ -f "$MIEWB_INST_DIR/miewb.env" ]; then
    while IFS= read -r _miewb_line || [ -n "$_miewb_line" ]; do
        case "$_miewb_line" in
            ''|\#*) continue ;;
        esac
        _miewb_key="${_miewb_line%%=*}"
        # trim surrounding whitespace from the key
        _miewb_key="$(printf '%s' "$_miewb_key" | tr -d '[:space:]')"
        case "$_miewb_key" in
            MIEWB_*) ;;
            *) continue ;;
        esac
        case "$_miewb_key" in
            *[!A-Z_]*) continue ;;
        esac
        _miewb_val="${_miewb_line#*=}"
        # trim leading/trailing whitespace (values are literal otherwise)
        _miewb_val="$(printf '%s' "$_miewb_val" \
            | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
        # env wins: export only if currently unset
        if eval "[ -z \"\${$_miewb_key+x}\" ]"; then
            eval "export $_miewb_key=\"\$_miewb_val\""
        fi
    done < "$MIEWB_INST_DIR/miewb.env"
    unset _miewb_line _miewb_key _miewb_val
elif [ "${MIEWB_ENV_QUIET:-}" != "1" ]; then
    echo "miewb_env.sh: $MIEWB_INST_DIR/miewb.env not found —" \
         "run scripts/setup_env.sh to create it" >&2
fi
