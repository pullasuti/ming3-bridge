# Ming-3.0-tts Wyoming bridge

Wyoming TTS server for Home Assistant, wrapping
[Ming-3.0-tts-0.5b](https://github.com/inclusionAI/Ming-3.0-tts) with zero-shot
voice cloning from a reference WAV and its transcript. Supports Wyoming
streaming (`synthesize-start` / `-chunk` / `-stop`), so Home Assistant can start
playback while the LLM is still writing.

## Quick start

```bash
cp .env.example .env      # or create .env, see below
./run.sh
```

`run.sh` creates a uv venv, clones Ming, installs dependencies (and a matching
flash-attn wheel if one exists), then starts the bridge on port 10212.

Minimal `.env`:

```
PROMPT_WAV=/path/to/voices/reference.wav
PROMPT_TEXT="Exact transcript of the reference audio."
EXTRA_ARGS="--cuda-graph --flow-steps 12"
```

| Variable | Default | Meaning |
|---|---|---|
| `PROMPT_WAV` / `PROMPT_TEXT` | - | Reference voice and its exact transcript. Without the transcript only the speaker embedding is used. |
| `MING_PORT` | 10212 | Wyoming port |
| `VOICE_SEED` | -1 | Fixed seed, or -1 for random |
| `PREBUFFER_SECONDS` | 1.5 | Audio buffered before playback starts |
| `MIN_MERGED_SECONDS` | 4.0 | Minimum size of a merged text block |
| `EXTRA_ARGS` | - | Extra bridge flags, e.g. `--cuda-graph --flow-steps 12` |
| `MING_DEBUG` | 0 | `1` for debug logging |
| `FLASH_ATTN` | 1 | `0` to skip flash-attn (gives no speedup for the 0.5b model) |

## CUDA graph: roughly 2.5-3x faster generation

Profiling showed the model is **CPU/launch-bound**, not GPU-bound: the GPU sat
at ~30% while one Python thread was pinned at 100%. About 70% of generation time
went into the flow-matching head, which launches many tiny kernels for every
audio patch.

`--cuda-graph` records that head once and replays it, removing the launch
overhead. Output is checked against eager sampling at capture time.

Real-time factor on an RTX 4070 (lower is better; below 1.0 keeps up with
playback), cloned voice:

| Flow steps | Default | `--cuda-graph` |
|---|---|---|
| 16 | 1.43 | 0.47 |
| 12 | 1.15 | **0.41** |
| 10 (model default) | 0.98 | 0.38 |
| 7 | 0.76 | 0.34 |
| 5 | 0.62 | 0.31 |

Without the graph, 10 steps barely kept up with realtime. With it, 12 steps run
at about 0.4x, so more flow steps (better quality) fit with headroom to spare.

`--flow-steps` accepts any number, but only 5, 6, 7, 10, 12 and 16 have tuned
timestep tables; other values use evenly spaced timesteps. If graph capture
fails, the bridge logs the error and restarts itself without `--cuda-graph`.

## Reference audio tips

- Ming continues from the **end** of the reference clip. Trim silence, room tone
  and fade-outs; `python prep_prompt.py voices/ref.wav -o voices/ref_trim.wav`
  reports and fixes this. A bad tail causes 0.3 s silent clips.
- The transcript must match the audio exactly and end with punctuation (a "."
  is added automatically if it is missing). 5-10 s of clean speech works well.

## Helper scripts

- `prep_prompt.py` - inspect / trim the reference WAV.
- `bench_steps.py` - compare RTF and audio across flow-step counts
  (`--cuda-graph`, `--builtin US_James`); WAVs are saved for listening.
