#!/usr/bin/env python3
"""Ming-3.0-tts Wyoming TTS bridge for Home Assistant.

In-process MingTTS with per-block task routing:
  - TTS task + prompt_wav + prompt_text  -> full-strength zero-shot clone
  - EMOTION task + prompt_wav only        -> discrete emotion, speaker-emb conditioning
  - inline non-verbal tags ride inside TTS blocks (canonical Ming casing)

Latency-first streaming: the FIRST ready sentence of each utterance is
generated and sent alone (fast time-to-first-audio); subsequent
sentences are merged into min_merged_seconds blocks for consistency.

Streaming guard (v2): when HA streams (synthesize-start / -chunk / -stop)
it ALSO sends a one-shot `synthesize` with the full text, purely for
backwards compatibility. A streaming server must ignore that one-shot.
v1 tried to detect it by comparing text, which broke whenever the two
strings differed slightly (style tags, spacing) and caused the stream to
be superseded and replayed. v2 uses an explicit `streaming` flag set
between synthesize-start and synthesize-stop. A short post-stop window
also suppresses a late duplicate one-shot with matching text.

Other changes in v2:
  - one shared single-thread executor on the engine (model is not
    thread-safe; previously each connection had its own pool)
  - tail sentence is only held back if the text does not already end
    in terminal punctuation (cuts first-audio latency by a full sentence)
  - real-time-factor logging per generated block
  - SynthesizeStopped is sent only for streaming utterances
  - epoch captured BEFORE any state reset can race with handlers
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncTcpServer
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
)

_LOGGER = logging.getLogger("bridge_ming")

# A one-shot synthesize arriving this soon after synthesize-stop, with
# matching text, is treated as a late duplicate of the stream.
LATE_DUPLICATE_WINDOW_S = 3.0

# Short-output detection: a block is "too short" if its audio is under
# MIN_SPEECH_RATIO of the estimated speaking time (floor MIN_SPEECH_FLOOR_S).
MIN_SPEECH_RATIO = 0.35
MIN_SPEECH_FLOOR_S = 0.45
MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Tag translation: CosyVoice-style lowercase -> Ming canonical (case-sensitive)
# ---------------------------------------------------------------------------

_CANONICAL = {
    "breath": "[Breathing]", "breathing": "[Breathing]",
    "cough": "[Cough]",
    "sigh": "[Sigh]",
    "laugh": "[Laughter]", "laughter": "[Laughter]",
    "surprise-ah": "[Surprise-ah]", "surprise-oh": "[Surprise-oh]",
    "dissatisfaction-hnn": "[Dissatisfaction-hnn]",
    "uhm": "[Uhm]", "um": "[Uhm]",
    "confirmation-en": "[Confirmation-en]",
    "question-ei": "[Question-ei]", "question-ah": "[Question-ah]",
    "question-en": "[Question-en]", "question-oh": "[Question-oh]",
    "question-huh": "[Question-huh]",
}

_INLINE_RE = re.compile(r"\[([^\[\]]+)\]")


def translate_inline_tags(text: str) -> str:
    def repl(m: re.Match) -> str:
        return _CANONICAL.get(m.group(1).lower().strip(), m.group(0))
    return _INLINE_RE.sub(repl, text)


def has_inline_tags(text: str) -> bool:
    return any(tag in text for tag in _CANONICAL.values())


# ---------------------------------------------------------------------------
# Emotion mapping: LLM style tags -> Ming's 7 discrete labels
# ---------------------------------------------------------------------------

EMOTION_MAP = {
    "warm": "高兴", "happy": "高兴", "cheerful": "高兴", "pleased": "高兴",
    "sad": "悲伤", "melancholic": "悲伤", "melancholy": "悲伤",
    "angry": "愤怒", "annoyed": "愤怒", "frustrated": "愤怒", "mad": "愤怒",
    "surprised": "惊讶", "shocked": "惊讶", "astonished": "惊讶",
    "fear": "恐惧", "afraid": "恐惧", "scared": "恐惧",
    "disgust": "厌恶", "disgusted": "厌恶",
    "neutral": "中性", "calm": "中性", "serious": "中性",
}


def pop_style_tag(text: str) -> tuple[str, str | None]:
    """Remove every style tag ([warm], [confident], ...) from text.

    Returns (clean_text, emotion). Non-verbal tags that Ming understands
    ([laugh], [breath], ...) stay in place. ALL other bracketed tags are
    removed, because Ming's text normalizer turns an unknown "[confident]"
    into the spoken word "confident". The first tag that maps to a
    non-neutral emotion in EMOTION_MAP is returned as the block emotion.
    """
    emotion: str | None = None

    def repl(m: re.Match) -> str:
        nonlocal emotion
        inner = m.group(1).lower().strip()
        if inner in _CANONICAL:
            return m.group(0)  # inline non-verbal tag, leave in place
        emo = EMOTION_MAP.get(inner)
        if emo and emo != "中性" and emotion is None:
            emotion = emo
        return " "

    return _INLINE_RE.sub(repl, text), emotion


def unknown_style_tags(text: str) -> list[str]:
    """Style tags that are neither non-verbal nor in EMOTION_MAP (for logging)."""
    out = []
    for m in _INLINE_RE.finditer(text):
        inner = m.group(1).lower().strip()
        if inner not in _CANONICAL and inner not in EMOTION_MAP:
            out.append(inner)
    return out


_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.!?;:…])")


def clean_spaces(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    return text.strip()


def append_chunk(buf: str, chunk: str) -> str:
    """Append a streamed text delta to the buffer.

    HA streams LLM token deltas that carry their own leading spaces
    ("twenty", "-", "fifth"), so deltas are concatenated as-is. Only when a
    delta starts a new sentence glued to the previous one ("Hello." then
    "How are you?") is a space inserted.
    """
    if not chunk:
        return buf
    if (buf and not buf[-1].isspace() and not chunk[0].isspace()
            and buf[-1] in ".!?…"
            and (chunk[0].isupper() or chunk[0] in "[\"'(")):
        return buf + " " + chunk
    return buf + chunk


def prepare_stream_text(buf: str, final: bool) -> tuple[str, str | None]:
    """Turn the raw streamed buffer into speakable text + emotion.

    Mid-stream, a partially received tag ("[conf") is held back so it is
    neither spoken nor dispatched half-formed.
    """
    if final:
        buf = re.sub(r"\[[^\[\]]*$", "", buf)  # unterminated tag at the end
    else:
        i = buf.rfind("[")
        if i != -1 and "]" not in buf[i:]:
            buf = buf[:i]
    text, emotion = pop_style_tag(buf)
    return clean_spaces(text), emotion


_TAG_SPLIT_RE = re.compile(r"(\[[^\[\]]*\])")
_WORD_HYPHEN_RE = re.compile(r"(?<=[A-Za-z])-(?=[A-Za-z])")


def fix_hyphens(text: str) -> str:
    """"twenty-fifth" -> "twenty fifth", leaving [Tag-names] alone.

    Ming's English text normalizer rewrites a hyphen between words as a
    full-width comma, so "twenty-fifth" is spoken with a pause in the middle.
    """
    parts = _TAG_SPLIT_RE.split(text)
    for i in range(0, len(parts), 2):  # even indices are outside [tags]
        parts[i] = _WORD_HYPHEN_RE.sub(" ", parts[i])
    return "".join(parts)


def _norm(text: str) -> str:
    """Whitespace/tag-insensitive lowercase text for duplicate comparison."""
    text = _INLINE_RE.sub(" ", text)
    return clean_spaces(text.lower())


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SynthEngine:
    def __init__(self, model_id: str, prompt_wav: str | None,
                 prompt_text: str | None,
                 voice_name: str = "nicoleheart",
                 dump_dir: str | None = None,
                 flow_steps: int = 0,
                 cuda_graph: bool = False) -> None:
        self.cuda_graph = bool(cuda_graph)
        self._graph_failed = False
        self.flow_steps = int(flow_steps or 0)  # 0 = model default (10)
        self.voice_name = voice_name
        self.dump_dir = dump_dir or None
        self._dump_n = 0
        self.model_id = model_id
        self.prompt_wav = prompt_wav
        # Ming glues the target text directly onto the reference transcript,
        # so the transcript must end with sentence punctuation or the first
        # word of every block fuses with its last word ("shutconfident").
        pt = (prompt_text or "").strip()
        if pt and pt[-1] not in ".!?…。！？":
            _LOGGER.warning(
                "prompt text does not end with punctuation; appending '.'")
            pt += "."
        self.prompt_text = pt
        self.model = None
        self.sample_rate = 44100
        # ONE worker shared by every connection: the model is not safe
        # to call concurrently.
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ming")

    def load(self) -> None:
        _LOGGER.info("loading %s ...", self.model_id)
        from ming_tts import MingTTS
        self.model = MingTTS(self.model_id)
        self.sample_rate = int(self.model.sample_rate)
        self._install_flow_steps_hook()
        _LOGGER.info("model ready (sample_rate=%d)", self.sample_rate)

    def _install_flow_steps_hook(self) -> None:
        """Wrap FlowLoss.sample: adjustable step count + optional CUDA graph.

        Profiling shows ~70% of generation time is the flow-matching head:
        `steps` (default 10) small DiT passes per audio patch, run as
        hundreds of tiny kernels from a single Python thread. The GPU sits
        ~70% idle waiting for launches.

        - self.flow_steps > 0 overrides the step count (Ming's decode loop
          never passes `steps`, so the model default is used otherwise).
        - self.cuda_graph captures the whole multi-step solve once per
          (steps, cfg, sigma, temperature, shapes) and replays it, removing
          the launch overhead. The random noise is still drawn outside the
          graph in the original order, so seeds behave the same.
        """
        import inspect
        try:
            flowloss = self.model.model.flowloss
            original = flowloss.sample
        except AttributeError as e:
            _LOGGER.warning("cannot hook flow-matching head (%s); "
                            "--flow-steps / --cuda-graph ignored", e)
            return
        try:
            default_steps = int(
                inspect.signature(original).parameters["steps"].default)
        except Exception:
            default_steps = 10
        engine = self
        graphs: dict = {}

        # fm.CFM.get_epss_timesteps builds its schedule with
        # torch.tensor(list, device="cuda"): a host->device copy, which CUDA
        # forbids while a graph is capturing (this is what broke the first
        # attempt). The schedule never changes, so build it once and reuse it.
        cfm_module = sys.modules.get(type(flowloss.cfm).__module__)
        epss = getattr(cfm_module, "get_epss_timesteps", None)
        if epss is not None and not getattr(epss, "_ming_cached", False):
            epss_cache: dict = {}

            def cached_epss(n, device, dtype):
                k = (n, str(device), dtype)
                if k not in epss_cache:
                    epss_cache[k] = epss(n, device, dtype)
                return epss_cache[k]

            cached_epss._ming_cached = True
            cfm_module.get_epss_timesteps = cached_epss

        def capture(key, z, latent_history, noise, sde_noise, cfg, patch_size,
                    sigma, temperature):
            import torch
            st = {"z": z.clone(), "h": latent_history.clone(),
                  "noise": noise.clone(), "sde": sde_noise.clone()}

            def run():
                return flowloss.cfm.sample(
                    noise=st["noise"], c=st["z"], latent_history=st["h"],
                    sde_noise=st["sde"], cfg_scale=cfg, patch_size=patch_size,
                    sigma=sigma, temperature=temperature)

            t0 = time.monotonic()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):  # warm-up on a side stream (required)
                    eager = run()
            torch.cuda.current_stream().wait_stream(side)
            eager = eager.clone()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = run()
            graph.replay()
            torch.cuda.synchronize()
            diff = (out.float() - eager.float()).abs().max().item()
            rel = diff / (eager.float().abs().max().item() + 1e-6)
            _LOGGER.info(
                "CUDA graph captured for steps=%d in %.1fs "
                "(max abs diff vs eager %.4g, relative %.4g)",
                key[0], time.monotonic() - t0, diff, rel)
            if not rel < 0.05:
                raise RuntimeError(
                    f"graph output differs from eager (relative {rel:.3g})")
            st["graph"], st["out"] = graph, out
            return st

        def sample(z, latent_history, cfg=1.0, patch_size=1, sigma=0.25,
                   temperature=0, steps=None):
            if steps is None:
                steps = engine.flow_steps if engine.flow_steps > 0 else default_steps
            if not engine.cuda_graph or engine._graph_failed:
                return original(z, latent_history, cfg, patch_size,
                                sigma=sigma, temperature=temperature,
                                steps=steps)
            import torch
            # same draw order as FlowLoss.sample, so a fixed seed matches
            noise = torch.randn((z.shape[0], patch_size, flowloss.z_channels),
                                device=z.device, dtype=z.dtype)
            sde_noise = torch.randn((steps, *noise.shape),
                                    device=z.device, dtype=z.dtype)
            key = (steps, float(cfg), int(patch_size), float(sigma),
                   float(temperature), tuple(z.shape),
                   tuple(latent_history.shape), z.dtype)
            entry = graphs.get(key)
            if entry is None:
                try:
                    entry = capture(key, z, latent_history, noise, sde_noise,
                                    cfg, patch_size, sigma, temperature)
                except Exception:
                    # A failed capture leaves CUDA's RNG state mid-capture, so
                    # even eager sampling is unusable in this process.
                    # warm_up() restarts the process without --cuda-graph.
                    engine._graph_failed = True
                    _LOGGER.exception("CUDA graph capture failed")
                    raise
                graphs[key] = entry
            entry["z"].copy_(z)
            entry["h"].copy_(latent_history)
            entry["noise"].copy_(noise)
            entry["sde"].copy_(sde_noise)
            entry["graph"].replay()
            return entry["out"].clone()

        flowloss.sample = sample
        if self.flow_steps > 0:
            _LOGGER.info("flow-matching steps: %d (model default %d)",
                         self.flow_steps, default_steps)
        if self.cuda_graph:
            _LOGGER.info("CUDA-graph sampling enabled (captured on first use)")

    def generate(self, text: str, emotion: str | None) -> np.ndarray:
        """Generate one block, retrying if the model returns near-silence.

        Ming's pipeline trims trailing silence and, when the whole clip is
        silent, returns a fixed 0.3s tail. So a ~0.30s result for a real
        sentence means the model generated silence and stopped after a few
        decode steps. Detect that by comparing against a rough expected
        duration and retry (the sampling is stochastic). The root-cause fix
        is a clean reference wav: see prep_prompt.py.
        """
        spoken = _INLINE_RE.sub(" ", text)
        expected = max(est_seconds(spoken) * MIN_SPEECH_RATIO, MIN_SPEECH_FLOOR_S)

        best = self._generate_once(text, emotion)
        attempt = 0
        while len(best) / float(self.sample_rate) < expected and attempt < MAX_RETRIES:
            attempt += 1
            _LOGGER.warning(
                "short/silent output (%.2fs, expected >= %.2fs) for %r; retry %d/%d",
                len(best) / float(self.sample_rate), expected, text[:60],
                attempt, MAX_RETRIES)
            pcm = self._generate_once(text, emotion)
            if len(pcm) > len(best):
                best = pcm

        if len(best) / float(self.sample_rate) < expected:
            _LOGGER.warning("still short after %d retries; using best result",
                            MAX_RETRIES)
        return best

    def _generate_once(self, text: str, emotion: str | None) -> np.ndarray:
        """One synthesis call. Returns int16 mono samples at sample_rate."""
        t0 = time.monotonic()
        text = fix_hyphens(translate_inline_tags(text))

        if emotion and not has_inline_tags(text):
            _LOGGER.info("EMOTION block (%s): %r", emotion, text[:60])
            waveform = self.model.synthesize(
                task="EMOTION", text=text,
                controls={"emotion": emotion},
                prompt_wav_path=self.prompt_wav,
                use_spk_emb=True,
            )
        else:
            if emotion:
                _LOGGER.info("inline tags present, emotion %s dropped for block",
                             emotion)
            if self.prompt_text:
                waveform = self.model.synthesize(
                    task="TTS", language_id="English", text=text,
                    prompt_wav_path=self.prompt_wav,
                    prompt_text=self.prompt_text,
                    use_spk_emb=True,
                )
            else:
                waveform = self.model.synthesize(
                    task="TTS", language_id="English", text=text,
                    prompt_wav_path=self.prompt_wav,
                    use_spk_emb=True,
                )

        samples = waveform.detach().cpu().numpy().squeeze().astype(np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=0)
        pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")

        gen_s = time.monotonic() - t0
        audio_s = len(pcm16) / float(self.sample_rate)
        if audio_s > 0:
            rtf = gen_s / audio_s
            _LOGGER.info(
                "generated %.2fs audio in %.2fs (RTF %.2f)%s",
                audio_s, gen_s, rtf,
                "  <-- slower than realtime, expect gaps"
                if rtf > 1.0 and audio_s >= 1.0 else "")
        if self.dump_dir:
            try:
                import soundfile as sf
                os.makedirs(self.dump_dir, exist_ok=True)
                self._dump_n += 1
                path = os.path.join(
                    self.dump_dir, f"{self._dump_n:04d}_{audio_s:.1f}s.wav")
                sf.write(path, pcm16, self.sample_rate)
                _LOGGER.info("dumped %s", path)
            except Exception as e:  # diagnostics must never break synthesis
                _LOGGER.warning("dump failed: %s", e)
        return pcm16

    def warm_up(self) -> None:
        _LOGGER.info("warm-up synthesis ...")
        try:
            t0 = time.monotonic()
            self._generate_once("Warm up.", emotion=None)
            _LOGGER.info("warm-up complete in %.1fs", time.monotonic() - t0)
        except Exception as e:
            if self._graph_failed:
                argv = [a for a in sys.argv if a != "--cuda-graph"]
                _LOGGER.error(
                    "CUDA graph capture failed during warm-up; restarting "
                    "without --cuda-graph (see the traceback above)")
                logging.shutdown()
                os.execv(sys.executable, [sys.executable] + argv)
            _LOGGER.warning("warm-up failed (non-fatal): %s", e)


# ---------------------------------------------------------------------------
# Text processing: sentence splitting + block merging
# ---------------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")
_ENDS_TERMINAL_RE = re.compile(r"[.!?…][\"')\]]*\s*$")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]


def ends_with_terminal(text: str) -> bool:
    return bool(_ENDS_TERMINAL_RE.search(text))


def est_seconds(sentence: str) -> float:
    return len(sentence.split()) / 2.5  # ~150 wpm


def merge_blocks(sentences: list[str], min_seconds: float) -> list[str]:
    blocks, cur, acc = [], [], 0.0
    for s in sentences:
        dur = est_seconds(s)
        if cur and acc + dur > min_seconds:
            blocks.append(" ".join(cur))
            cur, acc = [s], dur
        else:
            cur.append(s)
            acc += dur
    if cur:
        blocks.append(" ".join(cur))
    return blocks


# ---------------------------------------------------------------------------
# Prebuffering player: holds audio back until threshold, then streams freely
# ---------------------------------------------------------------------------

class PrebufferPlayer:
    """Emits ONE audio-start ... audio-stop sequence across its lifetime."""

    def __init__(self, handler: "MingEventHandler", prebuffer_seconds: float,
                 sample_rate: int, samples_per_chunk: int) -> None:
        self.handler = handler
        self.threshold = int(prebuffer_seconds * sample_rate) * 2  # bytes (i16)
        self.bytes_per_chunk = samples_per_chunk * 2
        self.buffer = bytearray()
        self.started = False
        self.pending = bytearray()
        self.finished = False
        self.abandoned = False  # set when superseded; all output stops
        self.t_start = time.monotonic()

    async def _start(self) -> None:
        _LOGGER.info("audio-start (after %.2fs)", time.monotonic() - self.t_start)
        await self.handler.write_event(
            AudioStart(self.handler.engine.sample_rate, 2, 1).event())
        self.started = True

    async def feed(self, pcm16: bytes) -> None:
        if self.finished or self.abandoned:
            return
        self.buffer.extend(pcm16)
        if not self.started:
            if len(self.buffer) < self.threshold:
                return
            await self._start()
        self.pending.extend(self.buffer)
        self.buffer.clear()
        while len(self.pending) >= self.bytes_per_chunk:
            if self.abandoned:
                return
            chunk = bytes(self.pending[:self.bytes_per_chunk])
            del self.pending[:self.bytes_per_chunk]
            await self.handler.write_event(AudioChunk(
                rate=self.handler.engine.sample_rate, width=2, channels=1,
                audio=chunk).event())

    async def finish(self) -> None:
        if self.finished or self.abandoned:
            return
        self.finished = True
        if not self.started:
            if not self.buffer:
                return  # nothing generated at all
            await self._start()
            self.pending.extend(self.buffer)
            self.buffer.clear()
        if self.pending:
            await self.handler.write_event(AudioChunk(
                rate=self.handler.engine.sample_rate, width=2, channels=1,
                audio=bytes(self.pending)).event())
            self.pending.clear()
        await self.handler.write_event(AudioStop().event())


# ---------------------------------------------------------------------------
# Wyoming handler
# ---------------------------------------------------------------------------

class MingEventHandler(AsyncEventHandler):
    def __init__(self, reader, writer, engine: SynthEngine,
                 min_merged_seconds: float, prebuffer_seconds: float,
                 samples_per_chunk: int) -> None:
        super().__init__(reader, writer)
        self.engine = engine
        self.min_merged_seconds = min_merged_seconds
        self.prebuffer_seconds = prebuffer_seconds
        self.samples_per_chunk = samples_per_chunk
        self.loop = asyncio.get_running_loop()

        # One utterance at a time per connection. `epoch` increments
        # whenever a new utterance starts; generation loops capture it at
        # entry and abort if it changes.
        self.epoch = 0
        self.busy = False

        # True between synthesize-start and synthesize-stop. While set,
        # any one-shot `synthesize` is HA's compat duplicate and is ignored.
        self.streaming = False

        # streaming synthesis state
        self.stream_buf = ""  # raw streamed text (token deltas concatenated)
        self._warned_tags: set[str] = set()
        self.stream_dispatched = 0  # sentences already handed to generation
        self.stream_style: str | None = None
        self.stream_player: PrebufferPlayer | None = None
        self.stream_has_audio = False

        # for late-duplicate suppression after synthesize-stop
        self.last_stream_norm = ""
        self.last_stream_end = 0.0

        _LOGGER.info("client connected from %s",
                     writer.get_extra_info("peername"))

    def _reset_stream_state(self) -> None:
        """Fresh state for a new utterance (increments epoch)."""
        self.epoch += 1
        if self.stream_player is not None and not self.stream_player.finished:
            # orphan any player from a previous, incomplete utterance so
            # its (possibly in-flight) generation writes nothing
            self.stream_player.abandoned = True
        self.stream_buf = ""
        self._warned_tags = set()
        self.stream_dispatched = 0
        self.stream_style = None
        self.stream_player = None
        self.stream_has_audio = False

    async def handle_event(self, event) -> bool:
        _LOGGER.info("event: %s", event.type)
        try:
            if Describe.is_type(event.type):
                await self.write_event(self._info().event())
                return True

            if Synthesize.is_type(event.type):
                # (1) Compat one-shot that HA sends mid-stream: ignore.
                if self.streaming:
                    syn_full = Synthesize.from_event(event)
                    _LOGGER.info(
                        "ignoring compat synthesize during streaming; "
                        "full text: %r", syn_full.text)
                    return True

                syn = Synthesize.from_event(event)
                text, emotion = pop_style_tag(syn.text or "")
                text = clean_spaces(text)

                # (2) Late duplicate right after a stream finished.
                if (self.last_stream_norm
                        and time.monotonic() - self.last_stream_end
                        < LATE_DUPLICATE_WINDOW_S
                        and _norm(text) == self.last_stream_norm):
                    _LOGGER.warning(
                        "late duplicate of finished stream suppressed: %r",
                        text[:60])
                    return True

                # (3) Genuine one-shot request.
                self._reset_stream_state()
                self.busy = True
                await self._run_utterance(text, emotion)
                return True

            if SynthesizeStart.is_type(event.type):
                _LOGGER.info("synthesize-start (streaming)")
                self._reset_stream_state()
                self.busy = True
                self.streaming = True
                self.stream_player = PrebufferPlayer(
                    self, self.prebuffer_seconds,
                    self.engine.sample_rate, self.samples_per_chunk)
                return True

            if SynthesizeChunk.is_type(event.type):
                if not self.streaming:
                    _LOGGER.debug("synthesize-chunk without start; ignored")
                    return True
                chunk = SynthesizeChunk.from_event(event)
                _LOGGER.debug("chunk text: %r", chunk.text)
                self.stream_buf = append_chunk(self.stream_buf, chunk.text or "")
                await self._flush_ready_blocks(final=False)
                return True

            if SynthesizeStop.is_type(event.type):
                _LOGGER.info("synthesize-stop (streaming)")
                if not self.streaming:
                    _LOGGER.debug("synthesize-stop without start; ignored")
                    return True
                my_epoch = self.epoch
                await self._flush_ready_blocks(final=True)
                if self.epoch == my_epoch:
                    player = self.stream_player
                    self.stream_player = None
                    if player is not None:
                        await player.finish()
                    self.last_stream_norm = _norm(self.stream_buf)
                    self.last_stream_end = time.monotonic()
                    self.streaming = False
                    self.busy = False
                    await self.write_event(SynthesizeStopped().event())
                return True

            # Unknown events (ping, etc.): KEEP the connection open
            _LOGGER.debug("ignoring event: %s", event.type)
            return True

        except (ConnectionResetError, BrokenPipeError):
            _LOGGER.warning("client disconnected during %s", event.type)
            if self.stream_player is not None:
                self.stream_player.abandoned = True
            return False

        except Exception:
            _LOGGER.exception("handler error on event type %s", event.type)
            try:
                await self._send_error("internal bridge error")
            except Exception:
                pass
            return False

    # -- single-shot utterance ------------------------------------------

    async def _run_utterance(self, text: str, emotion: str | None) -> None:
        my_epoch = self.epoch
        player = PrebufferPlayer(self, self.prebuffer_seconds,
                                 self.engine.sample_rate, self.samples_per_chunk)
        try:
            if text:
                blocks = merge_blocks(split_sentences(text),
                                      self.min_merged_seconds)
                for block in blocks:
                    _LOGGER.info("generating block: %r", block[:60])
                    pcm = await self.loop.run_in_executor(
                        self.engine.pool, self.engine.generate, block, emotion)
                    if self.epoch != my_epoch:
                        _LOGGER.info("utterance superseded; abandoning")
                        player.abandoned = True
                        return
                    await player.feed(pcm.tobytes())
            await player.finish()
        except (ConnectionResetError, BrokenPipeError):
            _LOGGER.warning("client disconnected during single-shot synthesis")
            player.abandoned = True
        except Exception:
            _LOGGER.exception("synthesis failed")
            await self._send_error("synthesis failed")
        finally:
            if self.epoch == my_epoch:
                self.busy = False

    # -- streaming utterance ----------------------------------------------

    async def _flush_ready_blocks(self, final: bool) -> None:
        my_epoch = self.epoch
        for tag in unknown_style_tags(self.stream_buf):
            if tag not in self._warned_tags:
                self._warned_tags.add(tag)
                _LOGGER.info(
                    "ignoring unmapped style tag [%s] (add it to EMOTION_MAP "
                    "to use it)", tag)
        text, emotion = prepare_stream_text(self.stream_buf, final)
        if emotion:
            self.stream_style = emotion
        sentences = split_sentences(text)

        # Hold back the last sentence only when it may still be growing,
        # i.e. the text so far does not end on terminal punctuation.
        if not final and sentences and not ends_with_terminal(text):
            sentences = sentences[:-1]

        new = sentences[self.stream_dispatched:]
        self.stream_dispatched = len(sentences)
        if not new:
            return

        if self.stream_has_audio:
            blocks = merge_blocks(new, self.min_merged_seconds)
        else:
            # LATENCY-FIRST: first ready sentence goes out alone.
            blocks = [new[0]] + merge_blocks(new[1:], self.min_merged_seconds)

        for block in blocks:
            _LOGGER.info("generating block: %r", block[:60])
            pcm = await self.loop.run_in_executor(
                self.engine.pool, self.engine.generate, block, self.stream_style)
            if self.epoch != my_epoch:
                _LOGGER.info("stream superseded; abandoning flush")
                return
            if self.stream_player is not None:
                await self.stream_player.feed(pcm.tobytes())
                if self.stream_player.started:
                    self.stream_has_audio = True

    async def _send_error(self, message: str) -> None:
        from wyoming.event import Event
        try:
            await self.write_event(Event("error", {"message": message}))
        except Exception:
            _LOGGER.exception("failed to send error event")

    def _info(self) -> Info:
        return Info(
            tts=[
                TtsProgram(
                    name="ming-3.0-tts",
                    attribution=Attribution(
                        name="Ming-3.0-tts",
                        url="https://github.com/inclusionAI/Ming-3.0-tts",
                    ),
                    installed=True,
                    description="Ming-3.0-tts-0.5B Wyoming bridge",
                    version="2.0.0",
                    voices=[
                        TtsVoice(
                            name=self.engine.voice_name,
                            attribution=Attribution(
                                name="Ming-3.0-tts",
                                url="https://github.com/inclusionAI/Ming-3.0-tts",
                            ),
                            installed=True,
                            description="cloned reference voice",
                            version="1.0.0",
                            languages=["en"],
                        )
                    ],
                    supports_synthesize_streaming=True,
                )
            ]
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(args) -> None:
    engine = SynthEngine(args.model_id, args.prompt_wav, args.prompt_text,
                         voice_name=args.voice, dump_dir=args.dump_dir,
                         flow_steps=args.flow_steps,
                         cuda_graph=args.cuda_graph)
    engine.load()
    if args.seed >= 0:
        import random
        import torch
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        _LOGGER.info("RNGs seeded with %d", args.seed)
    engine.warm_up()

    def handler_factory(reader, writer):
        return MingEventHandler(reader, writer, engine,
                                args.min_merged_seconds,
                                args.prebuffer_seconds,
                                args.samples_per_chunk)

    server = AsyncTcpServer("0.0.0.0", args.port)
    _LOGGER.info("serving on tcp://0.0.0.0:%d", args.port)
    await server.run(handler_factory)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=10213)
    ap.add_argument("--model-id", default="inclusionAI/Ming-3.0-tts-0.5b")
    ap.add_argument("--prompt-wav", required=True)
    ap.add_argument("--prompt-text", default="")
    ap.add_argument("--min-merged-seconds", type=float, default=4.0)
    ap.add_argument("--prebuffer-seconds", type=float, default=1.5)
    ap.add_argument("--samples-per-chunk", type=int, default=4096)
    ap.add_argument("--voice", default="nicoleheart")
    ap.add_argument("--flow-steps", type=int, default=0,
                    help="flow-matching steps per audio patch (0 = model default, 10). "
                         "Fewer is faster; judge quality by ear (try 7, 6, 5)")
    ap.add_argument("--cuda-graph", action="store_true",
                    help="capture the flow-matching solve as a CUDA graph "
                         "(removes CPU launch overhead; falls back to eager "
                         "if capture fails or the output differs)")
    ap.add_argument("--dump-dir", default="",
                    help="save every generated block as a WAV here (for listening)")
    ap.add_argument("--seed", type=int, default=-1,
                    help="seed RNGs once at startup (-1 = random)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
