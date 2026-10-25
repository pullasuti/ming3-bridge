#!/usr/bin/env python3
"""Compare speed and quality of different flow-matching step counts.

Run from ~/ming3 with the venv active and the bridge STOPPED (one GPU model
at a time is plenty):

    PYTHONPATH=Ming-3.0-tts python bench_steps.py \
        --prompt-wav voices/nicoleheart.wav --prompt-text "$PROMPT_TEXT"

Add --builtin US_James to use Ming's built-in voice instead of your reference
clip (task IP, no reference audio). That separates "my reference is the
problem" from "the model/settings are the problem".

For every step count it generates the same sentences with the same seed,
prints the real-time factor (RTF < 1.0 means faster than playback) and saves
WAVs to ./bench_out/ so you can listen and pick the lowest count that still
sounds good.
"""
import argparse
import os
import time

import numpy as np
import soundfile as sf
import torch

from bridge_ming import _INLINE_RE, SynthEngine

SENTENCES = [
    "It is currently six oh three in the evening on Sunday, October twenty-fifth.",
    "[breath] Is there anything else you need for this evening?",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="inclusionAI/Ming-3.0-tts-0.5b")
    ap.add_argument("--prompt-wav", default="")
    ap.add_argument("--builtin", default="",
                    help="built-in voice name, e.g. US_James (ignores --prompt-wav)")
    ap.add_argument("--prompt-text", default="")
    ap.add_argument("--steps", type=int, nargs="+", default=[16, 12, 10, 7, 6, 5])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--cuda-graph", action="store_true",
                    help="capture the flow-matching solve as a CUDA graph")
    ap.add_argument("--out", default="bench_out")
    args = ap.parse_args()

    if not args.builtin and not args.prompt_wav:
        ap.error("give --prompt-wav, or --builtin US_James")
    if args.prompt_wav and not args.builtin and not args.prompt_text.strip():
        print("WARNING: --prompt-text is empty, so Ming ignores the reference "
              "audio and only uses its speaker embedding. If you wrote "
              '--prompt-text "$PROMPT_TEXT", that shell variable is probably '
              "not set (run `source .env` or paste the transcript).\n")
    os.makedirs(args.out, exist_ok=True)
    eng = SynthEngine(args.model_id, args.prompt_wav, args.prompt_text)
    eng.load()
    eng.cuda_graph = args.cuda_graph

    def gen(text: str) -> np.ndarray:
        if not args.builtin:
            return eng._generate_once(text, None)
        text = _INLINE_RE.sub("", text).strip()  # IP task has no inline tags
        wav = eng.model.synthesize(
            task="IP", text=text, controls={"ip": args.builtin},
            use_zero_spk_emb=True)
        x = wav.detach().cpu().numpy().squeeze().astype(np.float32)
        return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")

    gen("Warm up.")  # first call is slow (CUDA init)

    print(f"{'steps':>5} {'sentence':>8} {'audio_s':>8} {'gen_s':>7} {'RTF':>6}")
    for steps in args.steps:
        eng.flow_steps = 0 if steps == 10 else steps
        gen("Warm up.")  # untimed: absorbs any CUDA-graph capture for this step count
        tot_audio = tot_gen = 0.0
        for i, text in enumerate(SENTENCES):
            torch.manual_seed(args.seed)
            t0 = time.monotonic()
            pcm = gen(text)
            gen_s = time.monotonic() - t0
            audio = len(pcm) / eng.sample_rate
            tot_audio += audio
            tot_gen += gen_s
            sf.write(os.path.join(args.out, f"{args.builtin or 'ref'}_steps{steps:02d}_s{i}.wav"),
                     pcm, eng.sample_rate)
            print(f"{steps:>5} {i:>8} {audio:>8.2f} {gen_s:>7.2f} {gen_s / max(audio, 1e-6):>6.2f}")
        print(f"{steps:>5} {'total':>8} {tot_audio:>8.2f} {tot_gen:>7.2f} "
              f"{tot_gen / max(tot_audio, 1e-6):>6.2f}\n")
    print(f"WAVs saved in {os.path.abspath(args.out)}  (compare steps10 vs the others by ear)")


if __name__ == "__main__":
    main()
