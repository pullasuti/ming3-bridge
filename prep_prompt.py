#!/usr/bin/env python3
"""Check (and optionally trim) the reference WAV used for Ming zero-shot cloning.

Why: Ming continues from the END of the prompt audio (its first decode step is
seeded from the prompt's tail). If the reference ends in silence, room tone or a
fade-out, the model tends to emit silence and stop after a few steps, which the
pipeline then trims to a fixed 0.3 s clip.

usage:
    python prep_prompt.py voices/nicoleheart.wav                 # report only
    python prep_prompt.py voices/nicoleheart.wav -o voices/nicoleheart_trim.wav
"""
import argparse

import numpy as np
import soundfile as sf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav")
    ap.add_argument("-o", "--out", help="write a trimmed copy here")
    ap.add_argument("--db", type=float, default=-18.0,
                    help="frames this many dB below the typical speech level "
                         "count as silence/fade (default -18)")
    ap.add_argument("--head", type=float, default=0.10,
                    help="seconds of lead-in to keep (default 0.10)")
    ap.add_argument("--tail", type=float, default=0.05,
                    help="seconds to keep after the last spoken frame (default 0.05)")
    args = ap.parse_args()

    x, sr = sf.read(args.wav, always_2d=True)
    mono = x.mean(axis=1)
    frame = max(1, int(sr * 0.02))  # 20 ms frames
    n = len(mono) // frame
    if n < 5:
        raise SystemExit("file too short")
    rms = np.sqrt((mono[: n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-12)

    # 1) rough speech mask -> typical speech level
    rough = rms > rms.max() * 10 ** (-45 / 20)
    typical = float(np.median(rms[rough]))
    # 2) real threshold relative to typical speech
    thr = typical * 10 ** (args.db / 20)
    voiced = np.where(rms > thr)[0]
    if len(voiced) == 0:
        raise SystemExit("no speech found above threshold")
    first, last = int(voiced[0]), int(voiced[-1])

    def db(v: float) -> float:
        return 20 * np.log10(max(v, 1e-9) / typical)

    dur = len(mono) / sr
    lead = first * frame / sr
    trail = (len(mono) - (last + 1) * frame) / sr
    tail_win = rms[max(0, n - int(0.3 / 0.02)):]
    print(f"file:            {args.wav}  ({sr} Hz, {x.shape[1]} ch)")
    print(f"duration:        {dur:.2f} s")
    print(f"leading silence: {lead:.2f} s")
    print(f"trailing silence:{trail:.2f} s")
    print(f"last 0.3 s level:{db(float(tail_win.mean())):+.1f} dB vs typical speech")

    problems = []
    if trail > 0.15:
        problems.append(f"ends with {trail:.2f}s of silence/room tone")
    if db(float(tail_win.mean())) < -12:
        problems.append("final 0.3 s is much quieter than the speech (fade-out?)")
    if dur > 15:
        problems.append(f"long reference ({dur:.0f}s); 5-10 s of clean speech is typical")
    if problems:
        print("\nLikely trouble for Ming (it continues from the end of this audio):")
        for p in problems:
            print("  -", p)
    else:
        print("\nTail looks fine.")

    if args.out:
        start = max(0, int(first * frame - args.head * sr))
        end = min(len(mono), int((last + 1) * frame + args.tail * sr))
        sf.write(args.out, x[start:end], sr)
        print(f"\nwrote {args.out}: {(end - start) / sr:.2f} s "
              f"(cut {start / sr:.2f}s head, {(len(mono) - end) / sr:.2f}s tail)")
        print("The transcript must still match the audio. If trimming cut into "
              "spoken words, trim the transcript to match.")


if __name__ == "__main__":
    main()
