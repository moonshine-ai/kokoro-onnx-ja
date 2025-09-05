import asyncio
import importlib
import importlib.metadata
import importlib.util
import os
import platform
import re
import time
from collections.abc import AsyncGenerator

import numpy as np
import onnxruntime as rt
from numpy.typing import NDArray

from misaki import ja

from .config import MAX_PHONEME_LENGTH, SAMPLE_RATE, KoKoroConfig, get_vocab
from .log import log
from .trim import trim as trim_audio


class Kokoro:
    def __init__(
        self,
        model_path: str,
        voice_path: str
    ):
        # Show useful information for bug reports
        log.debug(
            f"koko-onnx version {importlib.metadata.version('kokoro-onnx')} on {platform.platform()} {platform.version()}"
        )
        self.config = KoKoroConfig(model_path, voice_path)
        self.config.validate()

        # See list of providers https://github.com/microsoft/onnxruntime/issues/22101#issuecomment-2357667377
        providers = ["CPUExecutionProvider"]

        # Check if kokoro-onnx installed with kokoro-onnx[gpu] feature (Windows/Linux)
        gpu_enabled = importlib.util.find_spec("onnxruntime-gpu")
        if gpu_enabled:
            providers: list[str] = rt.get_available_providers()

        # Check if ONNX_PROVIDER environment variable was set
        env_provider = os.getenv("ONNX_PROVIDER")
        if env_provider:
            providers = [env_provider]

        log.debug(f"Providers: {providers}")
        self.sess = rt.InferenceSession(model_path, providers=providers)
        self.voice = np.fromfile(voice_path, dtype=np.float32).reshape(-1, 1, 256)

        g2p = ja.JAG2P()
        self.phonemize = lambda x: g2p(x)[0]
        self.vocab = get_vocab()

    def _tokenize(self, phonemes):
        if len(phonemes) > MAX_PHONEME_LENGTH:
            raise ValueError(
                f"text is too long, must be less than {MAX_PHONEME_LENGTH} phonemes"
            )
        return [i for i in map(self.vocab.get, phonemes) if i is not None]

    def _create_audio(
        self, phonemes: str, voice: NDArray[np.float32], speed: float
    ) -> tuple[NDArray[np.float32], int]:
        log.debug(f"Phonemes: {phonemes}")
        if len(phonemes) > MAX_PHONEME_LENGTH:
            log.warning(
                f"Phonemes are too long, truncating to {MAX_PHONEME_LENGTH} phonemes"
            )
        phonemes = phonemes[:MAX_PHONEME_LENGTH]
        start_t = time.time()
        tokens = np.array(self._tokenize(phonemes), dtype=np.int64)
        assert len(tokens) <= MAX_PHONEME_LENGTH, (
            f"Context length is {MAX_PHONEME_LENGTH}, but leave room for the pad token 0 at the start & end"
        )

        voice = voice[len(tokens)]
        tokens = [[0, *tokens, 0]]
        if "input_ids" in [i.name for i in self.sess.get_inputs()]:
            # Newer export versions
            inputs = {
                "input_ids": tokens,
                "style": np.array(voice, dtype=np.float32),
                "speed": np.array([speed], dtype=np.int32),
            }
        else:
            inputs = {
                "tokens": tokens,
                "style": voice,
                "speed": np.ones(1, dtype=np.float32) * speed,
            }

        audio = self.sess.run(None, inputs)[0]
        audio_duration = len(audio) / SAMPLE_RATE
        create_duration = time.time() - start_t
        rtf = create_duration / audio_duration
        log.debug(
            f"Created audio in length of {audio_duration:.2f}s for {len(phonemes)} phonemes in {create_duration:.2f}s (RTF: {rtf:.2f}"
        )
        return audio, SAMPLE_RATE

    def _split_phonemes(self, phonemes: str) -> list[str]:
        """
        Split phonemes into batches of MAX_PHONEME_LENGTH
        Prefer splitting at punctuation marks.
        """
        # Regular expression to split by punctuation and keep them
        words = re.split(r"([.,!?;])", phonemes)
        batched_phoenemes: list[str] = []
        current_batch = ""

        for part in words:
            # Remove leading/trailing whitespace
            part = part.strip()

            if part:
                # If adding the part exceeds the max length, split into a new batch
                # TODO: make it more accurate
                if len(current_batch) + len(part) + 1 >= MAX_PHONEME_LENGTH:
                    batched_phoenemes.append(current_batch.strip())
                    current_batch = part
                else:
                    if part in ".,!?;":
                        current_batch += part
                    else:
                        if current_batch:
                            current_batch += " "
                        current_batch += part

        # Append the last batch if it contains any phonemes
        if current_batch:
            batched_phoenemes.append(current_batch.strip())

        return batched_phoenemes

    def create(
        self,
        text: str,
        speed: float = 1.0,
        trim: bool = True,
    ) -> tuple[NDArray[np.float32], int]:
        """
        Create audio from text using the specified voice and speed.
        """
        assert speed >= 0.5 and speed <= 2.0, "Speed should be between 0.5 and 2.0"

        start_t = time.time()

        phonemes = self.phonemize(text)

        # Create batches of phonemes by splitting spaces to MAX_PHONEME_LENGTH
        batched_phoenemes = self._split_phonemes(phonemes)

        audio = []
        log.debug(
            f"Creating audio for {len(batched_phoenemes)} batches for {len(phonemes)} phonemes"
        )
        for phonemes in batched_phoenemes:
            audio_part, _ = self._create_audio(phonemes, self.voice, speed)
            if trim:
                # Trim leading and trailing silence for a more natural sound concatenation
                # (initial ~2s, subsequent ~0.02s)
                audio_part, _ = trim_audio(audio_part)
            audio.append(audio_part)
        audio = np.concatenate(audio)
        log.debug(f"Created audio in {time.time() - start_t:.2f}s")
        return audio, SAMPLE_RATE

    async def create_stream(
        self,
        text: str,
        speed: float = 1.0,
        trim: bool = True,
    ) -> AsyncGenerator[tuple[NDArray[np.float32], int], None]:
        """
        Stream audio creation asynchronously in the background, yielding chunks as they are processed.
        """
        assert speed >= 0.5 and speed <= 2.0, "Speed should be between 0.5 and 2.0"

        phonemes = self.phonemize(text)

        batched_phonemes = self._split_phonemes(phonemes)
        queue: asyncio.Queue[tuple[NDArray[np.float32], int] | None] = asyncio.Queue()

        async def process_batches():
            """Process phoneme batches in the background."""
            for i, phonemes in enumerate(batched_phonemes):
                loop = asyncio.get_event_loop()
                # Execute in separate thread since it's blocking operation
                audio_part, sample_rate = await loop.run_in_executor(
                    None, self._create_audio, phonemes, self.voice, speed
                )
                if trim:
                    # Trim leading and trailing silence for a more natural sound concatenation
                    # (initial ~2s, subsequent ~0.02s)
                    audio_part, _ = trim_audio(audio_part)
                log.debug(f"Processed chunk {i} of stream")
                await queue.put((audio_part, sample_rate))
            await queue.put(None)  # Signal the end of the stream

        # Start processing in the background
        asyncio.create_task(process_batches())

        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk
