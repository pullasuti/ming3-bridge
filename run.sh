#!/bin/bash
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Load .env if present
source .env 2>/dev/null || true

export MING_PORT="${MING_PORT:-10212}"
export PROMPT_WAV="${PROMPT_WAV:-$DIR/voices/nicoleheart.wav}"
export PROMPT_TEXT="${PROMPT_TEXT:-}"
export VOICE_SEED="${VOICE_SEED:--1}"
export VOICE_NAME="${VOICE_NAME:-nicoleheart}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

BRIDGE_SCRIPT="${BRIDGE_SCRIPT:-$DIR/bridge_ming.py}"

if [ ! -f "$BRIDGE_SCRIPT" ]; then
    echo "ERROR: bridge script not found: $BRIDGE_SCRIPT" >&2
    exit 1
fi

# uv bootstrap
if ! command -v uv &> /dev/null; then
    echo ">>> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# venv
if [ ! -d ".venv" ]; then
    echo ">>> Creating venv (Python 3.11)..."
    uv venv -p 3.11
fi
source .venv/bin/activate

# ---------------------------------------------------------------------------
# Ming-3.0-tts source (not pip-installable — run-in-place codebase)
# ---------------------------------------------------------------------------

MING_REPO="${MING_REPO:-$DIR/Ming-3.0-tts}"
MING_COMMIT="${MING_COMMIT:-028c1eace193e7acdb256453da89b6e9207e052b}"

if [ ! -d "$MING_REPO" ]; then
    echo ">>> Cloning Ming-3.0-tts ..."
    git clone https://github.com/inclusionAI/Ming-3.0-tts.git "$MING_REPO"
    git -C "$MING_REPO" checkout "$MING_COMMIT"
fi

# Expose ming_tts to the bridge
export PYTHONPATH="$MING_REPO${PYTHONPATH:+:$PYTHONPATH}"

if [ ! -f "$MING_REPO/ming_tts.py" ]; then
    echo "ERROR: ming_tts.py not found in $MING_REPO" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Dependencies (idempotent)
#
# torch is pinned to 2.8.x on the cu128 index because flash-attn 2.8.3 ships
# prebuilt wheels for torch 2.7/2.8 only. With unpinned torch on cu130 you get
# the newest torch, for which there is no flash-attn wheel and a source build
# tends to fail. cu128 builds run fine on newer (CUDA 13) drivers.
#
# To keep your old setup instead, put this in .env:
#   TORCH_PKGS="torch"
#   TORCH_INDEX="https://download.pytorch.org/whl/cu130"
#   FLASH_ATTN=0
#
# The stamp stores the torch settings, so changing them re-runs this block.
# ---------------------------------------------------------------------------

TORCH_PKGS="${TORCH_PKGS:-torch==2.8.0 torchaudio==2.8.0}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
DEPS_KEY="$TORCH_PKGS | $TORCH_INDEX"

if [ "$(cat .deps-stamp 2>/dev/null)" != "$DEPS_KEY" ]; then
    echo ">>> Installing/updating dependencies..."

    # The venv, pip-installable deps
    uv pip install -r "$MING_REPO/requirements.txt" \
        || echo "WARNING: some Ming requirements failed to install (continuing)" >&2
    # shellcheck disable=SC2086
    uv pip install $TORCH_PKGS --index-url "$TORCH_INDEX"
    uv pip install wyoming soundfile numpy pynini==2.1.6
    echo "$DEPS_KEY" > .deps-stamp
else
    echo ">>> Dependencies already installed (rm .deps-stamp to force)"
fi

# Show what we actually ended up with (requirements.txt can replace torch)
echo ">>> torch: $(python -c 'import torch; print(torch.__version__, "| cuda available:", torch.cuda.is_available())')"

# ---------------------------------------------------------------------------
# flash-attn (optional but recommended)
#
# The checkpoint pins 2 attention layers to flash-attn and silently falls back
# to SDPA without it. We only attempt the install when a prebuilt wheel should
# exist (torch 2.7/2.8) or you provide a wheel URL; otherwise a 20-30 minute
# source build would likely fail. A failure is remembered so it is not retried
# on every start (rm .flash-attn-failed to retry).
#
# Optional env: FLASH_ATTN=0 (disable), FLASH_ATTN_VERSION, FLASH_ATTN_WHEEL
# (URL or path to a wheel), FLASH_ATTN_FORCE_BUILD=1, MAX_JOBS.
# ---------------------------------------------------------------------------

FLASH_ATTN="${FLASH_ATTN:-1}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3}"

if [ "$FLASH_ATTN" = "1" ] && ! python -c "import flash_attn" 2>/dev/null; then
    TORCH_MM="$(python -c 'import torch; print(".".join(torch.__version__.split("+")[0].split(".")[:2]))')"

    if [ -f ".flash-attn-failed" ]; then
        echo ">>> flash-attn: earlier install failed; using SDPA (rm .flash-attn-failed to retry)" >&2
    elif [ -n "${FLASH_ATTN_WHEEL:-}" ] || [ "$TORCH_MM" = "2.7" ] \
         || [ "$TORCH_MM" = "2.8" ] || [ "${FLASH_ATTN_FORCE_BUILD:-0}" = "1" ]; then
        echo ">>> Installing flash-attn (torch $TORCH_MM) ..."
        uv pip install psutil ninja packaging setuptools wheel
        FA_SPEC="${FLASH_ATTN_WHEEL:-flash-attn==${FLASH_ATTN_VERSION}}"
        if MAX_JOBS="${MAX_JOBS:-4}" uv pip install --no-build-isolation --no-cache "$FA_SPEC" \
           && python -c "import flash_attn"; then
            echo ">>> flash-attn ready: $(python -c 'import flash_attn; print(flash_attn.__version__)')"
        else
            echo ">>> flash-attn install failed; continuing with SDPA" >&2
            touch .flash-attn-failed
        fi
    else
        echo ">>> flash-attn: no prebuilt wheel for torch $TORCH_MM; using SDPA." >&2
        echo "    Set TORCH_PKGS/TORCH_INDEX for torch 2.8 (default) or FLASH_ATTN_WHEEL=<wheel>." >&2
    fi
fi

# Sanity checks
if [ ! -f "$PROMPT_WAV" ]; then
    echo "WARNING: prompt WAV not found at $PROMPT_WAV" >&2
    echo "         cloning will fall back to default voice" >&2
fi
if [ -z "$PROMPT_TEXT" ]; then
    echo "WARNING: PROMPT_TEXT is not set — cloning uses speaker embedding only" >&2
    echo "         For full transcript conditioning, provide the exact reference transcript" >&2
fi

echo ">>> Starting Ming bridge on port $MING_PORT"
echo "    voice:   $VOICE_NAME ($PROMPT_WAV)"
echo "    transcript: ${PROMPT_TEXT:-not set (speaker-emb only)}"
echo "    seed:    $VOICE_SEED"
echo "    GPU:     CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

ARGS=(
    --port "$MING_PORT"
    --model-id "${MODEL_ID:-inclusionAI/Ming-3.0-tts-0.5b}"
    --prompt-wav "$PROMPT_WAV"
    --prompt-text "$PROMPT_TEXT"
    --voice "$VOICE_NAME"
    --seed "$VOICE_SEED"
    --min-merged-seconds "${MIN_MERGED_SECONDS:-4.0}"
    --prebuffer-seconds "${PREBUFFER_SECONDS:-1.5}"
    --samples-per-chunk "${SAMPLES_PER_CHUNK:-4096}"
)

# Debug logging is very noisy (torio, urllib3, text normalizer); opt in with MING_DEBUG=1
if [ "${MING_DEBUG:-0}" = "1" ]; then
    ARGS+=(--debug)
fi

# Free-form extra bridge flags, e.g. EXTRA_ARGS="--some-flag value"
# shellcheck disable=SC2206
EXTRA=(${EXTRA_ARGS:-})

exec python "$BRIDGE_SCRIPT" "${ARGS[@]}" "${EXTRA[@]}"
