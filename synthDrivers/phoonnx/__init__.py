import os
import sys
import threading
from collections import OrderedDict
from typing import OrderedDict as TOrderedDict, Optional, Set, Callable, List, Tuple
import queue

from nvwave import WavePlayer, AudioPurpose

from logHandler import log
from synthDriverHandler import (
    SynthDriver as BaseSynthDriver,
    VoiceInfo,
    synthIndexReached,
    synthDoneSpeaking,
)
from speech.commands import IndexCommand, PitchCommand, RateCommand, VolumeCommand, BreakCommand

try:
    import addonHandler
    addonHandler.initTranslation()
except Exception:
    _ = lambda s: s

DEFAULT_VOICE_ID = "dii_nl-NL"

DRIVER_DIR = os.path.dirname(os.path.abspath(__file__))
# Voices dropped here by the user are picked up without reinstalling the add-on.
VOICE_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "phoonnx", "voices")
PHOONNX_LIBS_PATH = os.path.join(DRIVER_DIR, "phoonnx_libs")
if PHOONNX_LIBS_PATH not in sys.path:
    sys.path.insert(0, PHOONNX_LIBS_PATH)

# How long speak() may wait for the async voice load before giving up.
VOICE_LOAD_TIMEOUT = 30.0
CHUNK_SIZE = 8192


class PhoonnxException(Exception):
    pass


def _voice_language(voice_id: str, config_path: str) -> Optional[str]:
    """Best-effort language for a voice: the config's lang code, else the
    ``name_ll-CC`` filename convention."""
    try:
        import json
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        for key in ("lang_code", "language", "lang"):
            value = cfg.get(key)
            if isinstance(value, str) and value:
                return value
            if isinstance(value, dict) and value.get("code"):
                return value["code"]
    except Exception:
        pass
    return voice_id.split("_")[-1] if "_" in voice_id else None


def discover_voices() -> "OrderedDict[str, dict]":
    """Find installed voices (``<id>.onnx`` + ``<id>.onnx.json`` pairs).

    Scans the add-on directory (bundled voices) and the phoonnx voice cache.
    Returns an ordered mapping of voice id to model/config paths + language,
    with the bundled default voice first when present.
    """
    voices: "OrderedDict[str, dict]" = OrderedDict()
    search_dirs = [DRIVER_DIR, VOICE_CACHE_DIR]
    for directory in search_dirs:
        if not os.path.isdir(directory):
            continue
        for fn in sorted(os.listdir(directory)):
            if not fn.endswith(".onnx"):
                continue
            voice_id = fn[:-len(".onnx")]
            model_path = os.path.join(directory, fn)
            config_path = model_path + ".json"
            if voice_id in voices or not os.path.exists(config_path):
                continue
            voices[voice_id] = {
                "model": model_path,
                "config": config_path,
                "language": _voice_language(voice_id, config_path),
            }
    if DEFAULT_VOICE_ID in voices:
        voices.move_to_end(DEFAULT_VOICE_ID, last=False)
    return voices


def nvda_rate_to_length_scale(rate: int) -> float:
    """Map the NVDA 0-100 rate to phoonnx ``length_scale``.

    Rate 50 is normal speed (1.0); the result is clamped to [0.2, 2.0] so a
    rate of 0 cannot divide by zero or produce an unusable scale.
    """
    rate = max(1, min(100, int(rate)))
    return max(0.2, min(2.0, 50.0 / rate))


def split_speech_sequence(speechSequence, initial_rate: int) -> Tuple[List[Tuple[str, object]], int]:
    """Flatten an NVDA speech sequence into ordered segments.

    Returns ``(segments, rate)`` where each segment is one of
    ``("text", str)``, ``("index", int)`` or ``("break", milliseconds)``.
    Index markers keep their position relative to the text so callers can
    notify ``synthIndexReached`` at the right time.
    """
    segments: List[Tuple[str, object]] = []
    rate = initial_rate
    pending_text = ""

    def flush():
        nonlocal pending_text
        if pending_text.strip():
            segments.append(("text", pending_text))
        pending_text = ""

    for item in speechSequence:
        if isinstance(item, str):
            pending_text += item
        elif isinstance(item, IndexCommand):
            flush()
            segments.append(("index", item.index))
        elif isinstance(item, BreakCommand):
            flush()
            segments.append(("break", item.time))
        elif isinstance(item, RateCommand):
            rate = item.value
        # PitchCommand/VolumeCommand are accepted but the model has no
        # per-utterance pitch control; volume is applied via SynthesisConfig.
    flush()
    return segments, rate


def import_phoonnx():
    """Import phoonnx lazily and return ``(PatchedVoice, SynthesisConfig)``.

    ``PatchedVoice`` wraps ``phoonnx.voice.TTSVoice`` with a
    ``synthesize_to_callback`` method that streams int16 audio chunks.
    """
    try:
        from phoonnx.config import SynthesisConfig
        from phoonnx.voice import TTSVoice as OriginalTTSVoice, LOG
        import numpy as np

        class PatchedVoice:
            """Wrapper around TTSVoice adding chunked callback synthesis."""

            def __init__(self, original_voice: OriginalTTSVoice):
                self._original_voice = original_voice

            @property
            def config(self):
                return self._original_voice.config

            @property
            def sample_rate(self):
                return self._original_voice.config.sample_rate

            @property
            def phonetic_spellings(self):
                return self._original_voice.phonetic_spellings

            @property
            def phonemizer(self):
                return self._original_voice.phonemizer

            def phonemize(self, text: str):
                return self._original_voice.phonemize(text)

            def phonemes_to_ids(self, phonemes):
                return self._original_voice.phonemes_to_ids(phonemes)

            def phoneme_ids_to_audio(self, phoneme_ids, config):
                return self._original_voice.phoneme_ids_to_audio(phoneme_ids, config)

            @staticmethod
            def load(model_path, config_path):
                original_voice = OriginalTTSVoice.load(model_path, config_path)
                return PatchedVoice(original_voice)

            def synthesize_to_callback(self,
                                       text: str,
                                       audio_callback: Callable,
                                       config: Optional[SynthesisConfig] = None,
                                       speaker_id: Optional[int] = None) -> bool:
                """Synthesize ``text`` and stream int16 chunks to ``audio_callback``.

                ``audio_callback`` returns nonzero to abort. Returns True on
                success, False on error or abort.
                """
                if config is None:
                    config = SynthesisConfig()
                try:
                    if self.phonetic_spellings and config.enable_phonetic_spellings:
                        text = self.phonetic_spellings.apply(text)
                    if config.add_diacritics:
                        text = self.phonemizer.add_diacritics(text, self.config.lang_code)

                    sentence_phonemes = self.phonemize(text)
                    for phonemes in sentence_phonemes:
                        if not phonemes:
                            continue
                        phoneme_ids = self.phonemes_to_ids(phonemes)
                        if not phoneme_ids:
                            continue

                        audio_float_array = self.phoneme_ids_to_audio(phoneme_ids, config)

                        max_val = np.max(np.abs(audio_float_array))
                        if max_val >= 1e-8:
                            audio_float_array = audio_float_array / max_val
                        if config.volume != 1.0:
                            audio_float_array = audio_float_array * config.volume

                        audio_float_array = np.clip(audio_float_array, -1.0, 1.0).astype(np.float32)
                        audio_int16_bytes: bytes = (audio_float_array * 32767).astype(np.int16).tobytes()

                        for i in range(0, len(audio_int16_bytes), CHUNK_SIZE):
                            if audio_callback(audio_int16_bytes[i:i + CHUNK_SIZE]):
                                return False
                    return True
                except Exception as e:
                    LOG.error(f"PatchedVoice: Error during synthesis: {e}")
                    return False

        log.info("Phoonnx: TTSVoice and dependencies successfully imported (Wrapped).")
        return PatchedVoice, SynthesisConfig

    except (ImportError, ModuleNotFoundError, AttributeError) as e:
        log.critical(f"FATAL ERROR: Failed to load Phoonnx or dependency. Check bundling: {e}", exc_info=True)
        return None, None


class _VoiceLoaderThread(threading.Thread):
    """Asynchronously loads the TTSVoice instance and the WavePlayer."""

    def __init__(self, driver: 'SynthDriver', voice_id: str, model_path: str, config_path: str,
                 generation: int = 0):
        super().__init__()
        self.driver = driver
        self.voice_id = voice_id
        self.model_path = model_path
        self.config_path = config_path
        self.generation = generation
        self.daemon = True

    def run(self):
        log.info(f"Phoonnx ASYNC: Starting asynchronous loading of voice '{self.voice_id}'...")
        try:
            VoiceClass, SynthesisConfigClass = import_phoonnx()
            if VoiceClass is None or SynthesisConfigClass is None:
                raise PhoonnxException("TTS modules could not be loaded.")

            tts_voice = VoiceClass.load(self.model_path, self.config_path)
            samplesPerSec = getattr(tts_voice, "sample_rate", 22050)

            player = WavePlayer(
                channels=1,
                samplesPerSec=samplesPerSec,
                bitsPerSample=16,
                purpose=AudioPurpose.SPEECH
            )

            if self.generation != self.driver._load_generation:
                # A newer voice switch superseded this load; discard it.
                player.close()
                return

            old_player = self.driver._player
            self.driver.tts_voice = tts_voice
            self.driver._player = player
            self.driver._SynthesisConfig_class = SynthesisConfigClass
            if old_player is not None and old_player is not player:
                try:
                    old_player.close()
                except Exception:
                    pass
            log.info("Phoonnx ASYNC: Loading of TTSVoice and WavePlayer complete.")

        except Exception as e:
            log.error(f"Phoonnx ASYNC: Failed to load voice '{self.voice_id}': {e}", exc_info=True)
            if self.generation == self.driver._load_generation:
                self.driver.tts_voice = None
                self.driver._player = None
                self.driver._SynthesisConfig_class = None
        finally:
            if self.generation == self.driver._load_generation:
                self.driver._voice_loaded_event.set()


class _SynthQueueThread(threading.Thread):
    """Permanent worker thread that processes speech requests from the queue."""

    def __init__(self, driver: 'SynthDriver'):
        super().__init__()
        self.driver = driver
        self.daemon = True
        self.stop_event = threading.Event()
        self.cancel_synthesis_event = threading.Event()

    def _aborted(self) -> bool:
        return self.stop_event.is_set() or self.cancel_synthesis_event.is_set()

    def run(self):
        log.info("Phoonnx SynthQueueThread: Starting permanent processing loop.")

        while not self.stop_event.is_set():
            try:
                # request is: (segments, config, index_callback, player_ref, tts_voice)
                request = self.driver._request_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if self.stop_event.is_set():
                self.driver._request_queue.task_done()
                break

            try:
                segments, length_scale, volume, index_callback = request
                self.cancel_synthesis_event.clear()

                # First request may arrive while the voice is still loading.
                self.driver._voice_loaded_event.wait(VOICE_LOAD_TIMEOUT)
                tts_voice = self.driver.tts_voice
                player_ref = self.driver._player
                config_cls = self.driver._SynthesisConfig_class
                if tts_voice is None or player_ref is None or config_cls is None:
                    log.warning("Phoonnx: Voice not available; dropping request.")
                    index_callback(None)
                    continue
                config = config_cls(
                    length_scale=length_scale,
                    enable_phonetic_spellings=True,
                    volume=volume,
                )

                def thread_local_audio_callback(chunk: bytes) -> int:
                    if self._aborted():
                        return 1
                    try:
                        player_ref.feed(chunk)
                    except Exception:
                        return 1
                    return 0

                sample_rate = getattr(tts_voice, "sample_rate", 22050)
                for kind, value in segments:
                    if self._aborted():
                        break
                    if kind == "text":
                        tts_voice.synthesize_to_callback(
                            value,
                            config=config,
                            audio_callback=thread_local_audio_callback,
                        )
                    elif kind == "index":
                        index_callback(value)
                    elif kind == "break":
                        n_samples = int(sample_rate * (value / 1000.0))
                        if n_samples > 0:
                            thread_local_audio_callback(bytes(n_samples * 2))

                if not self._aborted():
                    try:
                        # Flush buffered audio so the tail of the utterance plays.
                        player_ref.idle()
                    except Exception:
                        pass
                index_callback(None)

            except Exception as e:
                if not self._aborted():
                    log.error(f"Phoonnx Error (QueueThread): TTS synthesis failed: {e}", exc_info=True)
            finally:
                self.driver._request_queue.task_done()

        log.info("Phoonnx SynthQueueThread: Processing loop has stopped.")

    def cancel_synthesis(self):
        self.cancel_synthesis_event.set()


class SynthDriver(BaseSynthDriver):
    """NVDA SynthDriver implementation for the Phoonnx TTS engine."""

    name = "phoonnx"
    description = _("Phoonnx TTS Driver")

    supportedCommands = frozenset([IndexCommand, RateCommand, VolumeCommand, BreakCommand])
    supportedNotifications = frozenset([synthIndexReached, synthDoneSpeaking])

    supportedSettings = (
        BaseSynthDriver.VoiceSetting(),
        BaseSynthDriver.RateSetting(),
        BaseSynthDriver.VolumeSetting(),
    )

    _availableVoicesCache: Optional[TOrderedDict[str, VoiceInfo]] = None
    _rate: int = 50
    _volume: int = 100

    def __init__(self):
        super(SynthDriver, self).__init__()
        self.tts_voice: Optional[object] = None
        self._voice_id: Optional[str] = None
        self._player: Optional[WavePlayer] = None
        self._SynthesisConfig_class: Optional[type] = None

        self._request_queue: queue.Queue = queue.Queue()
        self._worker_thread: Optional[_SynthQueueThread] = None

        self._voice_loaded_event = threading.Event()
        self._loader_thread: Optional[_VoiceLoaderThread] = None
        self._load_generation: int = 0
        self._voice_catalog = discover_voices()

        if self.check():
            self._get_voice()
            self._worker_thread = _SynthQueueThread(driver=self)
            self._worker_thread.start()

    @classmethod
    def check(cls) -> bool:
        if not discover_voices():
            log.warning("Phoonnx check failed: no voice (.onnx + .onnx.json pair) found "
                        f"in {DRIVER_DIR} or {VOICE_CACHE_DIR}.")
            return False
        return True

    def _getAvailableVoices(self) -> TOrderedDict[str, VoiceInfo]:
        if self._availableVoicesCache is None:
            self._availableVoicesCache = OrderedDict()
            for voice_id, info in self._voice_catalog.items():
                display_name = f"Phoonnx ({voice_id.replace('_', ' ')})"
                self._availableVoicesCache[voice_id] = VoiceInfo(
                    voice_id, display_name, language=info["language"])
        return self._availableVoicesCache

    def _get_voice(self) -> Optional[str]:
        if self._voice_id is None and self._voice_catalog:
            self._voice_id = next(iter(self._voice_catalog))
            self._load_tts_voice()
        return self._voice_id

    def _set_voice(self, value: str):
        if value not in self._voice_catalog:
            log.warning(f"Phoonnx: Attempting to set invalid voice: {value}.")
            return
        if self._voice_id != value:
            self._voice_id = value
            self.cancel()
            self._load_tts_voice()

    def _load_tts_voice(self):
        info = self._voice_catalog.get(self._voice_id)
        if info is None:
            return
        log.info(f"Phoonnx: Starting asynchronous loading for voice '{self._voice_id}'.")
        self._load_generation += 1
        self._voice_loaded_event.clear()
        self._loader_thread = _VoiceLoaderThread(
            driver=self,
            voice_id=self._voice_id,
            model_path=info["model"],
            config_path=info["config"],
            generation=self._load_generation,
        )
        self._loader_thread.start()

    def _get_rate(self) -> int:
        return self._rate

    def _set_rate(self, value: int):
        self._rate = value

    def _get_volume(self) -> int:
        return self._volume

    def _set_volume(self, value: int):
        self._volume = value

    def _get_language(self) -> Optional[str]:
        voice_info = self.availableVoices.get(self._get_voice())
        return voice_info.language if voice_info else None

    def _set_language(self, language):
        pass

    def _get_availableLanguages(self) -> Set[Optional[str]]:
        return {v.language for v in self.availableVoices.values()}

    def _onIndexReached(self, index: Optional[int]):
        if index is not None:
            synthIndexReached.notify(synth=self, index=index)
        else:
            synthDoneSpeaking.notify(synth=self)

    def speak(self, speechSequence):
        """Queue a speech request; never blocks the NVDA main thread."""
        if self._worker_thread is None:
            log.warning("Phoonnx: Cannot speak, driver failed check().")
            return
        if self._voice_loaded_event.is_set() and (
                not self.tts_voice or not self._player or not self._SynthesisConfig_class):
            log.warning("Phoonnx: Cannot speak, TTS voice, WavePlayer, or Config Class not loaded.")
            return

        segments, current_rate = split_speech_sequence(speechSequence, self._get_rate())
        if not segments:
            return

        length_scale = nvda_rate_to_length_scale(current_rate)
        self._request_queue.put(
            (segments, length_scale, self._volume / 100.0, self._onIndexReached)
        )

    def cancel(self):
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.cancel_synthesis_event.set()
        while not self._request_queue.empty():
            try:
                self._request_queue.get(block=False)
                self._request_queue.task_done()
            except queue.Empty:
                break
        if self._player:
            self._player.stop()

    def pause(self, switch: bool):
        if self._player:
            self._player.pause(switch)

    def terminate(self):
        log.info("Phoonnx: Driver is terminating.")
        self.cancel()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.stop_event.set()
            self._worker_thread.join(timeout=1)
            if self._worker_thread.is_alive():
                log.warning("Phoonnx: QueueThread did not shut down within 1 second. Continuing.")
        if self._player:
            self._player.close()
            self._player = None
        self.tts_voice = None
        self._SynthesisConfig_class = None
