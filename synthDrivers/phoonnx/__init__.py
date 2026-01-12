import os
import sys
import threading
from collections import OrderedDict
from typing import OrderedDict as TOrderedDict, Optional, Set, Callable, Dict, Any, List
import queue
import time
import json 
from pathlib import Path 
import os.path

# --- PATH CONFIGURATION ---
# We determine the paths for the addon and its bundled libraries.
DRIVER_DIR = os.path.dirname(os.path.abspath(__file__))
ADDON_ROOT_DIR = os.path.dirname(os.path.dirname(DRIVER_DIR))

# Ensure the addon root and library folder are in sys.path.
if ADDON_ROOT_DIR not in sys.path:
    sys.path.insert(0, ADDON_ROOT_DIR)

PHOONNX_LIBS_PATH = os.path.join(ADDON_ROOT_DIR, "phoonnx_libs")
if PHOONNX_LIBS_PATH not in sys.path:
    # We insert at index 0 to ensure our bundled libraries take priority over 
    # other versions potentially installed in the global environment.
    sys.path.insert(0, PHOONNX_LIBS_PATH)

# --- CRITICAL FIX: MANUAL MODULE INJECTION ---
# In NVDA's shared Python environment, 'dateutil' and other libraries often fail 
# due to lazy loading or conflicts with other addons.
# We force-load and register these submodules into sys.modules to prevent ModuleNotFoundError.
try:
    # Force load dateutil submodules required by dateparser/ovos_date_parser.
    import dateutil
    import dateutil.relativedelta
    import dateutil.parser
    sys.modules['dateutil.relativedelta'] = dateutil.relativedelta
    sys.modules['dateutil.parser'] = dateutil.parser
    
    # Register common dependencies to ensure they are available globally within NVDA.
    import pytz
    sys.modules['pytz'] = pytz
    
    import six
    sys.modules['six'] = six
    
    import dateparser
    sys.modules['dateparser'] = dateparser
except Exception:
    # If injection fails, we continue silently; errors will be caught during phoonnx imports.
    pass

# --- THIRD-PARTY IMPORTS ---
# After path configuration and injection, it is safe to import heavy dependencies.
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

class PhoonnxException(Exception): pass

def load_voice_configs() -> Dict[str, Any]:
    """Scans the voices directory and loads parameters directly from the voice's model.json."""
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
                
                configs[voice_id] = {
                    "display_name": voice_id.split('/')[-1].replace("pipertts_", ""),
                    "language": meta.get("language", {}).get("code", "en"),
                    "onnx_path": os.path.join(root, "model.onnx"),
                    "config_path": config_path,
                    "sample_rate": sample_rate,
                    "inference": meta.get("inference", {})
                }
                log.info(f"Phoonnx: Voice loaded: {voice_id} at {sample_rate}Hz")
    except Exception as e:
        log.error(f"Phoonnx: Error scanning voices: {e}")
    return configs

class PatchedVoice:
    """Wrapper for the core TTSVoice to handle NVDA-specific audio streaming and normalization."""
    def __init__(self, original_voice: TTSVoice, sample_rate: int):
        self._original_voice = original_voice
        self._sample_rate = sample_rate
    
    @property
    def sample_rate(self): 
        return self._sample_rate

    def synthesize_to_callback(self, text: str, audio_callback: Callable, index_callback: Callable, config: Optional[SynthesisConfig] = None):
        if config is None: config = SynthesisConfig()
        try:
            if self._original_voice.phonetic_spellings and config.enable_phonetic_spellings:
                text = self._original_voice.phonetic_spellings.apply(text)
            
            sentence_phonemes = self._original_voice.phonemize(text)
            
            for phonemes in sentence_phonemes:
                if not phonemes: continue
                phoneme_ids = self._original_voice.phonemes_to_ids(phonemes)
                audio_float_array = self._original_voice.phoneme_ids_to_audio(phoneme_ids, config)
                
                # Normalize and clip audio for safe playback
                max_val = np.max(np.abs(audio_float_array))
                if max_val > 0.0001: 
                    audio_float_array = audio_float_array / max_val
                
                effective_vol = max(config.volume, 0.05)
                audio_float_array = np.clip(audio_float_array * effective_vol, -1.0, 1.0).astype(np.float32)
                
                # Convert to 16-bit PCM bytes for NVWave
                audio_int16_bytes = (audio_float_array * 32767).astype(np.int16).tobytes()
                
                CHUNK_SIZE = 1024 
                for i in range(0, len(audio_int16_bytes), CHUNK_SIZE):
                    chunk = audio_int16_bytes[i:i + CHUNK_SIZE]
                    if chunk:
                        audio_callback(chunk)
            
            index_callback(None)
        except Exception as e:
            log.error(f"PatchedVoice Error: {e}")
            index_callback(None)

class _VoiceLoaderThread(threading.Thread):
    """Handles async loading of ONNX models to prevent freezing the NVDA UI."""
    def __init__(self, driver, voice_id, paths):
        super().__init__()
        self.driver, self.voice_id, self.paths = driver, voice_id, paths
        self.daemon = True

    def run(self):
        try:
            original_voice = TTSVoice.load(self.paths['onnx_path'], self.paths['config_path'])
            tts_voice = PatchedVoice(original_voice, self.paths.get('sample_rate', 22050))
            
            s_rate = tts_voice.sample_rate
            player = WavePlayer(
                channels=1, 
                samplesPerSec=s_rate, 
                bitsPerSample=16
            )
            
            self.driver.tts_voice = tts_voice
            self.driver._player = player
        except Exception as e:
            log.error(f"Phoonnx: Loading {self.voice_id} failed: {e}")

class _QueueThread(threading.Thread):
    """Processes speech requests sequentially from a queue."""
    def __init__(self, request_queue):
        super().__init__()
        self.request_queue = request_queue
        self.stop_event = threading.Event()
        self.cancel_synthesis_event = threading.Event()
        self.daemon = True

    def run(self):
        while not self.stop_event.is_set():
            try:
                text, config, index_callback, player, tts_voice = self.request_queue.get(timeout=0.5)
                self.cancel_synthesis_event.clear()

                def audio_callback(chunk):
                    if not self.cancel_synthesis_event.is_set() and player: 
                        player.feed(chunk)

                def internal_index_callback(index):
                    if index is None: synthDoneSpeaking.notify()
                    index_callback(index)

                tts_voice.synthesize_to_callback(text, audio_callback, internal_index_callback, config=config)
                self.request_queue.task_done()
            except queue.Empty:
                continue

class SynthDriver(BaseSynthDriver):
    """NVDA Synth Driver implementation for Phoonnx TTS."""
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
        self._worker_thread = _QueueThread(self._request_queue)
        self._worker_thread.start()

        # Initialize config with integer validation to prevent string-related errors
        if "phoonnx" not in config.conf["speech"]:
            config.conf["speech"]["phoonnx"] = {}
        
        try:
            current_rate = int(config.conf["speech"]["phoonnx"].get("rate", 50))
        except (ValueError, TypeError):
            current_rate = 50
        config.conf["speech"]["phoonnx"]["rate"] = current_rate

        try:
            current_vol = int(config.conf["speech"]["phoonnx"].get("volume", 100))
        except (ValueError, TypeError):
            current_vol = 100
        config.conf["speech"]["phoonnx"]["volume"] = current_vol
        
        last_voice = config.conf["speech"]["phoonnx"].get("voice")
        if last_voice and last_voice in self._voice_configs:
            self.voice = last_voice
        elif self._voice_configs:
            self.voice = list(self._voice_configs.keys())[0]

    def _get_availableVoices(self) -> OrderedDict[str, VoiceInfo]:
        """Returns the list of voices and ensures a refresh when called."""
        self._voice_configs = load_voice_configs()
        voices = OrderedDict()
        if not self._voice_configs:
            voices["no_voices_found"] = VoiceInfo("no_voices_found", "No voices found", "en")
            return voices
        for v_id, cfg in self._voice_configs.items():
            voices[v_id] = VoiceInfo(v_id, cfg['display_name'], cfg['language'])
        return voices

    def updateVoiceList(self):
        """Forces NVDA to reload the voice list from the UI."""
        from speech import voicesChanged
        voicesChanged.notify()

    def _get_voice(self):
        return self._current_voice_id

    def _set_voice(self, voice_id):
        if not voice_id or voice_id not in self._voice_configs:
            return
        self._current_voice_id = voice_id
        paths = self._voice_configs[voice_id]
        if self._player:
            self._player.stop()
            self._player.close()
            self._player = None
        self.tts_voice = None
        _VoiceLoaderThread(self, voice_id, paths).start()

    def _get_rate(self):
        try:
            return int(config.conf["speech"]["phoonnx"].get("rate", 50))
        except:
            return 50

    def _set_rate(self, value):
        config.conf["speech"]["phoonnx"]["rate"] = int(value)

    def _get_volume(self):
        try:
            return int(config.conf["speech"]["phoonnx"].get("volume", 100))
        except:
            return 100

    def _set_volume(self, value):
        config.conf["speech"]["phoonnx"]["volume"] = int(value)

    def speak(self, speechSequence):
        """Processes the speech sequence and calculates synthesis parameters."""
        if not self.tts_voice or self._current_voice_id not in self._voice_configs: return
        
        text = "".join([item for item in speechSequence if isinstance(item, str)])
        if not text.strip(): return
        
        voice_meta = self._voice_configs[self._current_voice_id]
        inference = voice_meta.get("inference", {})
        
        try:
            current_rate = int(self.rate)
        except:
            current_rate = 50
            
        base_ls = inference.get("length_scale", 1.0)
        nvda_rate = max(current_rate, 1)
        
        # Calculate length_scale based on NVDA rate
        length_scale = base_ls * (1.5 - (nvda_rate / 100.0) * 1.2)
        length_scale = max(0.2, min(length_scale, 2.5))
        
        try:
            current_vol = int(self.volume)
        except:
            current_vol = 100
        vol_float = float(current_vol) / 100.0
        
        synthesis_config = SynthesisConfig(
            length_scale=float(length_scale),
            noise_scale=inference.get("noise_scale", 0.667),
            noise_w_scale=inference.get("noise_w", 0.8),
            volume=vol_float,
            enable_phonetic_spellings=True
        )
        
        self._request_queue.put((text, synthesis_config, self._onIndexReached, self._player, self.tts_voice))

    def _onIndexReached(self, index):
        if index is not None: synthIndexReached.notify(index)

    def cancel(self):
        """Immediately stops the player and clears the pending speech queue."""
        if self._player: self._player.stop()
        self._worker_thread.cancel_synthesis_event.set()
        while not self._request_queue.empty():
            try: 
                self._request_queue.get(block=False)
                self._request_queue.task_done()
            except queue.Empty: break

    def pause(self, switch: bool):
        if self._player: self._player.pause(switch)

    def terminate(self):
        """Cleans up threads and player before the driver is unloaded."""
        if self._player: self._player.close()
        self._worker_thread.stop_event.set()
        self._worker_thread.join(timeout=1)
