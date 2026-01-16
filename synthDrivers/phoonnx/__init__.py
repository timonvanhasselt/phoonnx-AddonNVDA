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

if ADDON_ROOT_DIR not in sys.path:
    sys.path.insert(0, ADDON_ROOT_DIR)

PHOONNX_LIBS_PATH = os.path.join(ADDON_ROOT_DIR, "phoonnx_libs")
if PHOONNX_LIBS_PATH not in sys.path:
    sys.path.insert(0, PHOONNX_LIBS_PATH)

# --- CRITICAL FIX: MANUAL MODULE INJECTION ---
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

# --- THIRD-PARTY IMPORTS ---
import numpy as np 
from nvwave import WavePlayer, AudioPurpose 
import config

# --- NVDA CORE IMPORTS ---
from logHandler import log
from synthDriverHandler import (
    SynthDriver as BaseSynthDriver,
    VoiceInfo,
    synthIndexReached,
    synthDoneSpeaking,
)
from speech.commands import IndexCommand, PitchCommand, RateCommand, VolumeCommand, BreakCommand
_ = lambda s: s

# --- PHOONNX SPECIFIC IMPORTS ---
from phoonnx.voice import TTSVoice
from phoonnx.config import SynthesisConfig

# --- DIRECTORY SETUP ---
USER_HOME = os.path.expanduser("~")
PHOONNX_CACHE_DIR = os.path.join(USER_HOME, ".cache", "phoonnx")
VOICES_ROOT = os.path.join(PHOONNX_CACHE_DIR, "voices")

def chunk_text(text: str, max_len: int = 250) -> List[str]:
    if not text:
        return []
    # Splitst op leestekens gevolgd door een spatie om natuurlijke zinsgrenzen te behouden
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
                    "inference": meta.get("inference", {})
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
                    # Alleen synthetiseren als er tekst is
                    if text.strip():
                        chunks = chunk_text(text)
                        for chunk in chunks:
                            if self.cancel_event.is_set(): break
                            tts_voice.synthesize_to_callback(chunk, audio_callback, config=config)

                    # Wacht tot de audio fysiek klaar is met afspelen voor dit tekstblok
                    if player and not self.cancel_event.is_set():
                        player.sync() 

                    # Meld de index aan NVDA nadat de audio klaar is. 
                    # Dit houdt de visuele cursor synchroon bij 'Alles Lezen'.
                    if last_index is not None and not self.cancel_event.is_set():
                        synthIndexReached.notify(synth=self.driver, index=last_index)

                finally:
                    # Laat NVDA weten dat we klaar zijn met dit item in de sequence
                    synthDoneSpeaking.notify(synth=self.driver)
                    self.request_queue.task_done()
            except queue.Empty:
                continue

class SynthDriver(BaseSynthDriver):
    name = "phoonnx"
    description = "Phoonnx TTS"
    supportedSettings = (BaseSynthDriver.VoiceSetting(), BaseSynthDriver.RateSetting(), BaseSynthDriver.VolumeSetting())

    @classmethod
    def check(cls): return True

    def __init__(self):
        super().__init__()
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
        # Ook als er alleen een index is zonder tekst (bijv. witregels), 
        # sturen we dit naar de queue voor correcte afhandeling.
        if not combined_text.strip() and last_index is None: 
            return

        inference = self._voice_configs[self._current_voice_id].get("inference", {})
        length_scale = inference.get("length_scale", 1.0) * (1.5 - (self.rate / 100.0) * 1.2)
        s_config = SynthesisConfig(
            length_scale=max(0.2, min(length_scale, 2.5)),
            noise_scale=inference.get("noise_scale", 0.667),
            noise_w_scale=inference.get("noise_w", 0.8),
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
