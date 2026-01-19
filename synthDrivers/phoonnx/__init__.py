import os
import sys
import threading
from collections import OrderedDict
from typing import OrderedDict as TOrderedDict, Optional, Set, Callable, Dict, Any, List
import queue
import time
import json 
import re
from pathlib import Path 
import os.path

# --- PATH CONFIGURATION ---
DRIVER_DIR = os.path.dirname(os.path.abspath(__file__))
ADDON_ROOT_DIR = os.path.dirname(os.path.dirname(DRIVER_DIR))

# --- ESPEAK-NG LIBRARIES CONFIGURATION ---
BIN_DIR = os.path.join(ADDON_ROOT_DIR, "bin")
ESPEAK_EXE = os.path.join(BIN_DIR, "espeak-ng.exe")
ESPEAK_DATA = os.path.join(BIN_DIR, "espeak-ng-data")

if os.path.isdir(BIN_DIR):
    os.environ["PATH"] = BIN_DIR + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(BIN_DIR)
        except Exception:
            pass
    os.environ["ESPEAK_DATA_PATH"] = BIN_DIR  
    os.environ["PHOONNX_ESPEAK_EXECUTABLE"] = ESPEAK_EXE
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = os.path.join(BIN_DIR, "libespeak-ng.dll")
    os.environ["PHONEMIZER_ESPEAK_PATH"] = BIN_DIR

if ADDON_ROOT_DIR not in sys.path:
    sys.path.insert(0, ADDON_ROOT_DIR)

PHOONNX_LIBS_PATH = os.path.join(ADDON_ROOT_DIR, "phoonnx_libs")
if PHOONNX_LIBS_PATH not in sys.path:
    sys.path.insert(0, PHOONNX_LIBS_PATH)

try:
    import dateutil
    import dateutil.relativedelta
    import dateutil.parser
    sys.modules['dateutil.relativedelta'] = dateutil.relativedelta
    sys.modules['dateutil.parser'] = dateutil.parser
    import pytz
    sys.modules['pytz'] = pytz
    import six
    sys.modules['six'] = six
    import dateparser
    sys.modules['dateparser'] = dateparser
except Exception:
    pass

import numpy as np 
from nvwave import WavePlayer, AudioPurpose 
import config

from logHandler import log
from synthDriverHandler import (
    SynthDriver as BaseSynthDriver,
    VoiceInfo,
    synthIndexReached,
    synthDoneSpeaking,
)
from speech.commands import IndexCommand, PitchCommand, RateCommand, VolumeCommand, BreakCommand
_ = lambda s: s

from phoonnx.voice import TTSVoice
from phoonnx.config import SynthesisConfig

USER_HOME = os.path.expanduser("~")
PHOONNX_CACHE_DIR = os.path.join(USER_HOME, ".cache", "phoonnx")
VOICES_ROOT = os.path.join(PHOONNX_CACHE_DIR, "voices")

def chunk_text(text: str, max_len: int = 250) -> List[str]:
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_len:
            current_chunk += " " + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def load_voice_configs() -> Dict[str, Any]:
    configs = {}
    try:
        if not os.path.exists(VOICES_ROOT):
            os.makedirs(VOICES_ROOT, exist_ok=True)
        for root, dirs, files in os.walk(VOICES_ROOT):
            if "model.onnx" in files and "model.json" in files:
                rel_path = os.path.relpath(root, VOICES_ROOT)
                voice_id = rel_path.replace("\\", "/")
                config_path = os.path.join(root, "model.json")
                with open(config_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                sample_rate = meta.get("audio", {}).get("sample_rate", 22050)
                raw_name = voice_id.split('/')[-1]
                display_name = raw_name.replace("pipertts_", "").replace("_", " ").replace("-", " ").capitalize()
                configs[voice_id] = {
                    "display_name": display_name,
                    "language": meta.get("language", {}).get("code", "en"),
                    "onnx_path": os.path.join(root, "model.onnx"),
                    "config_path": config_path,
                    "sample_rate": sample_rate,
                    "inference": meta.get("inference", {}),
                    "speaker_id_map": meta.get("speaker_id_map", {}),
                    "num_speakers": meta.get("num_speakers", 1)
                }
    except Exception as e:
        log.error(f"Phoonnx: Error scanning voices: {e}")
    return configs

class PatchedVoice:
    def __init__(self, original_voice: TTSVoice, sample_rate: int):
        self._original_voice = original_voice
        self._sample_rate = sample_rate
    
    @property
    def sample_rate(self): 
        return self._sample_rate

    def synthesize_to_callback(self, text: str, audio_callback: Callable, config: Optional[SynthesisConfig] = None):
        if config is None: config = SynthesisConfig()
        try:
            if self._original_voice.phonetic_spellings and config.enable_phonetic_spellings:
                text = self._original_voice.phonetic_spellings.apply(text)
            
            if BIN_DIR not in os.environ["PATH"]:
                os.environ["PATH"] = BIN_DIR + os.pathsep + os.environ.get("PATH", "")

            sentence_phonemes = self._original_voice.phonemize(text)
            for phonemes in sentence_phonemes:
                if not phonemes: continue
                phoneme_ids = self._original_voice.phonemes_to_ids(phonemes)
                audio_float_array = self._original_voice.phoneme_ids_to_audio(phoneme_ids, config)
                max_val = np.max(np.abs(audio_float_array))
                if max_val > 0.0001: 
                    audio_float_array = audio_float_array / max_val
                effective_vol = max(config.volume, 0.05)
                audio_float_array = np.clip(audio_float_array * effective_vol, -1.0, 1.0).astype(np.float32)
                audio_int16_bytes = (audio_float_array * 32767).astype(np.int16).tobytes()
                audio_callback(audio_int16_bytes)
        except Exception as e:
            log.error(f"PatchedVoice Error: {e}")

class _VoiceLoaderThread(threading.Thread):
    def __init__(self, driver, voice_id, paths):
        super().__init__()
        self.driver, self.voice_id, self.paths = driver, voice_id, paths
        self.daemon = True

    def run(self):
        try:
            original_voice = TTSVoice.load(self.paths['onnx_path'], self.paths['config_path'])
            tts_voice = PatchedVoice(original_voice, self.paths.get('sample_rate', 22050))
            player = WavePlayer(channels=1, samplesPerSec=tts_voice.sample_rate, bitsPerSample=16)
            self.driver.tts_voice = tts_voice
            self.driver._player = player
        except Exception as e:
            log.error(f"Phoonnx: Loading failed: {e}")

class _QueueThread(threading.Thread):
    def __init__(self, driver, request_queue):
        super().__init__()
        self.driver = driver
        self.request_queue = request_queue
        self.stop_event = threading.Event()
        self.cancel_event = threading.Event()
        self.daemon = True

    def run(self):
        while not self.stop_event.is_set():
            try:
                item = self.request_queue.get(timeout=0.1)
                text, config, last_index, player, tts_voice = item
                self.cancel_event.clear()

                def audio_callback(chunk):
                    if not self.cancel_event.is_set() and player: 
                        player.feed(chunk)

                try:
                    if text.strip():
                        chunks = chunk_text(text)
                        for chunk in chunks:
                            if self.cancel_event.is_set(): break
                            tts_voice.synthesize_to_callback(chunk, audio_callback, config=config)

                    if player and not self.cancel_event.is_set():
                        player.sync() 

                    if last_index is not None and not self.cancel_event.is_set():
                        synthIndexReached.notify(synth=self.driver, index=last_index)

                finally:
                    synthDoneSpeaking.notify(synth=self.driver)
                    self.request_queue.task_done()
            except queue.Empty:
                continue

class SynthDriver(BaseSynthDriver):
    name = "phoonnx"
    description = "Phoonnx TTS"

    @classmethod
    def check(cls): return True

    def __init__(self):
        super().__init__()
        log.info(f"Phoonnx: Initializing. Bin path: {BIN_DIR}")
        
        if not os.path.exists(ESPEAK_EXE):
            log.error(f"Phoonnx: espeak-ng.exe NOT found at {ESPEAK_EXE}")

        self._current_voice_id = "" 
        self._voice_configs = load_voice_configs()
        self.tts_voice = self._player = None
        self._request_queue = queue.Queue()
        self._worker_thread = _QueueThread(self, self._request_queue)
        self._worker_thread.start()

        if "phoonnx" not in config.conf["speech"]:
            config.conf["speech"]["phoonnx"] = {}
        
        self.rate = int(config.conf["speech"]["phoonnx"].get("rate", 50))
        self.volume = int(config.conf["speech"]["phoonnx"].get("volume", 100))
        
        last_voice = config.conf["speech"]["phoonnx"].get("voice")
        if last_voice and last_voice in self._voice_configs:
            self.voice = last_voice
        elif self._voice_configs:
            self.voice = list(self._voice_configs.keys())[0]

    @property
    def supportedSettings(self):
        """Dynamically determine supported settings based on the current voice."""
        settings = [
            BaseSynthDriver.VoiceSetting(),
            BaseSynthDriver.RateSetting(),
            BaseSynthDriver.VolumeSetting()
        ]
        
        # Check if the current voice has multiple speakers
        cfg = self._voice_configs.get(self._current_voice_id)
        if cfg and cfg.get("speaker_id_map") and len(cfg["speaker_id_map"]) > 1:
            settings.insert(1, BaseSynthDriver.VariantSetting())
            
        return tuple(settings)

    def _get_availableVoices(self) -> OrderedDict[str, VoiceInfo]:
        self._voice_configs = load_voice_configs()
        voices = OrderedDict()
        if not self._voice_configs:
            voices["no_voices_found"] = VoiceInfo("no_voices_found", "No voices found", "en")
            return voices
        for v_id, cfg in sorted(self._voice_configs.items()):
            voices[v_id] = VoiceInfo(v_id, cfg['display_name'], cfg['language'])
        return voices

    def _get_voice(self): return self._current_voice_id
    def _set_voice(self, voice_id):
        if not voice_id or voice_id not in self._voice_configs: return
        self._current_voice_id = voice_id
        config.conf["speech"]["phoonnx"]["voice"] = voice_id
        if self._player:
            self._player.stop()
            self._player.close()
            self._player = None
        self.tts_voice = None
        _VoiceLoaderThread(self, voice_id, self._voice_configs[voice_id]).start()

    def _get_availableVariants(self) -> OrderedDict[str, VoiceInfo]:
        variants = OrderedDict()
        cfg = self._voice_configs.get(self._current_voice_id)
        if not cfg or not cfg.get("speaker_id_map"):
            variants["0"] = VoiceInfo("0", _("Default"))
            return variants
        
        sorted_speakers = sorted(cfg["speaker_id_map"].items(), key=lambda x: x[1])
        for name, s_id in sorted_speakers:
            variants[str(s_id)] = VoiceInfo(str(s_id), str(s_id))
        return variants

    def _get_variant(self):
        return str(config.conf["speech"]["phoonnx"].get("variant", "0"))

    def _set_variant(self, value):
        config.conf["speech"]["phoonnx"]["variant"] = str(value)

    def _get_rate(self): return int(config.conf["speech"]["phoonnx"].get("rate", 50))
    def _set_rate(self, value): config.conf["speech"]["phoonnx"]["rate"] = int(value)
    def _get_volume(self): return int(config.conf["speech"]["phoonnx"].get("volume", 100))
    def _set_volume(self, value): config.conf["speech"]["phoonnx"]["volume"] = int(value)

    def speak(self, speechSequence):
        if not self.tts_voice: return
        text_parts = []
        last_index = None
        for item in speechSequence:
            if isinstance(item, str): 
                text_parts.append(item)
            elif isinstance(item, IndexCommand): 
                last_index = item.index
        
        combined_text = "".join(text_parts)
        if not combined_text.strip() and last_index is None: 
            return

        inference = self._voice_configs[self._current_voice_id].get("inference", {})
        length_scale = inference.get("length_scale", 1.0) * (1.5 - (self.rate / 100.0) * 1.2)
        
        try:
            speaker_id = int(config.conf["speech"]["phoonnx"].get("variant", 0))
        except (ValueError, TypeError):
            speaker_id = 0

        s_config = SynthesisConfig(
            length_scale=max(0.2, min(length_scale, 2.5)),
            noise_scale=inference.get("noise_scale", 0.667),
            noise_w_scale=inference.get("noise_w", 0.8),
            speaker_id=speaker_id,
            volume=float(self.volume) / 100.0,
            enable_phonetic_spellings=True
        )
        self._request_queue.put((combined_text, s_config, last_index, self._player, self.tts_voice))

    def cancel(self):
        self._worker_thread.cancel_event.set()
        if self._player: self._player.stop()
        while not self._request_queue.empty():
            try: 
                self._request_queue.get(block=False)
                self._request_queue.task_done()
            except queue.Empty: break

    def pause(self, switch: bool):
        if self._player: self._player.pause(switch)

    def terminate(self):
        if self._player: self._player.close()
        self._worker_thread.stop_event.set()
        self._worker_thread.join(timeout=1)
