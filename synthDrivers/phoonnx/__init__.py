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
from nvwave import WavePlayer, AudioPurpose 
import numpy as np 
import config

# --- Essential NVDA Core Imports ---
from logHandler import log
from synthDriverHandler import (
    SynthDriver as BaseSynthDriver,
    VoiceInfo,
    synthIndexReached,
    synthDoneSpeaking,
)
from speech.commands import IndexCommand, PitchCommand, RateCommand, VolumeCommand, BreakCommand
_ = lambda s: s

# --- CRUCIAL CONFIGURATION ---
VOICE_CONFIG_FILE = "voices.json"
DRIVER_DIR = os.path.dirname(os.path.abspath(__file__))

log.debug("PHOONNX DEBUG: __init__.py has started execution.")

# Path calculation for the root of the add-on (Two levels up from synthDrivers/phoonnx)
ADDON_ROOT_DIR = os.path.dirname(os.path.dirname(DRIVER_DIR))

# --- Python Search Path Configuration (CRUCIAL) ---

# 1. Add the root of the add-on (to be able to import 'phoonnx' and voiceSelector)
if ADDON_ROOT_DIR not in sys.path:
    sys.path.insert(0, ADDON_ROOT_DIR)

# 2. Add the separate libs folder (if used for other dependencies)
PHOONNX_LIBS_PATH = os.path.join(ADDON_ROOT_DIR, "phoonnx_libs")
if PHOONNX_LIBS_PATH not in sys.path:
    sys.path.insert(0, PHOONNX_LIBS_PATH)

# IMPORTS FOR FIRST-RUN LOGIC
from . import voiceSelector 
# MODIFIED: Use 'initDone' as the key, as requested.
FIRST_RUN_KEY = "initDone" # Key for NVDA config

# --- Global Exception Definition ---
class PhoonnxException(Exception): pass

# --- FUNCTION TO LOAD VOICE CONFIGURATIONS (ADJUSTED) ---
def load_voice_configs() -> Dict[str, Dict[str, str]]:
    """
    Reads the available voice configurations from the JSON file and scans the
    local installation directory for flat and nested filenames.
    """
    configs = {}
    config_path = os.path.join(DRIVER_DIR, VOICE_CONFIG_FILE)
    
    # --- 1. Load from voices.json ---
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                configs = json.load(f)
                log.info(f"Phoonnx: Successfully loaded {len(configs)} voice configurations from JSON.")
        except json.JSONDecodeError as e:
            log.critical(f"FATAL ERROR: Invalid JSON in voice configuration file: {e}")
        except Exception as e:
            log.critical(f"FATAL ERROR: Failed to read voice configuration file: {e}", exc_info=True)

    # --- 2. Scan the Local Download Directory (VOICE_INSTALL_DIR) for flat and nested files ---
    VOICE_INSTALL_DIR = Path(os.path.join(ADDON_ROOT_DIR, 'synthDrivers', 'phoonnx', 'voices'))
    DRIVER_PATH = Path(DRIVER_DIR)
    
    if VOICE_INSTALL_DIR.is_dir():
        log.info(f"Phoonnx: Starting to scan for flat and nested voice files in: {VOICE_INSTALL_DIR}")
        
        # 2a. Flat files in voices/ (e.g. dii_nl-NL.onnx)
        flat_model_files = list(VOICE_INSTALL_DIR.glob("*.onnx")) + list(VOICE_INSTALL_DIR.glob("*.pt"))
        
        # 2b. Nested files (recursive) (e.g. voices/OpenVoiceOS/pipertts_nl-NL_miro/model.onnx)
        # We look for 'model.onnx' and 'model.pt' in subdirectories.
        nested_model_files = list(VOICE_INSTALL_DIR.rglob("model.onnx")) + list(VOICE_INSTALL_DIR.rglob("model.pt"))
        
        all_model_files = flat_model_files + nested_model_files
        
        for model_file in all_model_files:
            
            # Determine the structure and paths
            if model_file.parent == VOICE_INSTALL_DIR:
                # Structure: voices/dii_nl-NL.onnx (Flat)
                voice_id = model_file.stem
                config_file = model_file.with_suffix(model_file.suffix + '.json') 
                
                # Relative paths: voices/dii_nl-NL.onnx
                relative_model_path = os.path.join('voices', model_file.name)
                relative_config_path = os.path.join('voices', config_file.name)
                
            else:
                # Structure: voices/.../VoiceID/model.onnx (Nested)
                # The voice ID is the name of the directory containing model.onnx
                voice_id = model_file.parent.name
                config_file = model_file.parent / "model.json" 
                
                # Relative paths: from DRIVER_DIR (synthDrivers/phoonnx)
                # Example: voices/OpenVoiceOS/pipertts_nl-NL_miro/model.onnx
                relative_model_path = str(model_file.relative_to(DRIVER_PATH)).replace('\\', '/')
                relative_config_path = str(config_file.relative_to(DRIVER_PATH)).replace('\\', '/')
                
            # Check if the configuration file exists
            if config_file.exists():
                
                # Deduction of the language ID
                parts = voice_id.split('_')
                if len(parts) > 1:
                    # Use the part after the first underscore, e.g., 'nl-NL'
                    lang_tag = '_'.join(parts[1:]) 
                else:
                    lang_tag = 'und' 
                
                if voice_id not in configs:
                    # Add the locally downloaded/installed voice
                    configs[voice_id] = {
                        "display_name": f"{voice_id} (Local/Auto-detect)", 
                        "language": lang_tag,
                        "model_file": relative_model_path,
                        "config_file": relative_config_path
                    }
                    log.info(f"Phoonnx: Local voice '{voice_id}' dynamically added. Path: {relative_model_path}. Language: {lang_tag}")
                    
                else:
                    # Overwrite the paths if the voice is already in voices.json
                    configs[voice_id]['model_file'] = relative_model_path
                    configs[voice_id]['config_file'] = relative_config_path
                    log.info(f"Phoonnx: Voice '{voice_id}' from JSON UPDATED to local path: {relative_model_path}")


    if not configs:
         log.critical("FATAL ERROR: No valid voice configurations found.")

    return configs


def import_phoonnx():
    """
    Imports Phoonnx and returns the wrapped TTSVoice class AND SynthesisConfig.
    """
    try:
        from phoonnx.config import SynthesisConfig
        from phoonnx.voice import TTSVoice as OriginalTTSVoice, LOG
        
        class PatchedVoice:
            """
            Wrapper around OriginalTTSVoice with the required synthesize_to_callback.
            """

            def __init__(self, original_voice: OriginalTTSVoice):
                self._original_voice = original_voice

            # --- Essential Properties and Methods to forward ---
            @property
            def config(self):
                return self._original_voice.config

            @property
            def sample_rate(self):
                # or self._original_voice.sample_rate if available
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

            # --- Static Method: Loads the Wrapper ---
            @staticmethod
            def load(model_path, config_path):
                # Load the original voice and wrap it
                original_voice = OriginalTTSVoice.load(model_path, config_path)
                return PatchedVoice(original_voice)

            # --- The Custom Callback Logic ---
            def synthesize_to_callback(self,
                                       text: str,
                                       audio_callback: Callable,
                                       index_callback: Callable,
                                       config: Optional[SynthesisConfig] = None,
                                       speaker_id: Optional[int] = None):

                if config is None: config = SynthesisConfig()
                LOG.debug("text=%s", text)
                try:

                    if self.phonetic_spellings and config.enable_phonetic_spellings:
                        text = self.phonetic_spellings.apply(text)
                    if config.add_diacritics:
                        text = self.phonemizer.add_diacritics(text, self.config.lang_code)

                    sentence_phonemes = self.phonemize(text)
                    all_phoneme_ids_for_synthesis = [
                        self.phonemes_to_ids(phonemes) for phonemes in sentence_phonemes if phonemes
                    ]

                    first_chunk = True
                    sentence_silence = 0.0
                    silence_int16_bytes = bytes(
                        int(self.config.sample_rate * sentence_silence * 2)
                    )

                    for phoneme_ids in all_phoneme_ids_for_synthesis:
                        if not phoneme_ids: continue
                        if not first_chunk: audio_callback(silence_int16_bytes)
                        first_chunk = False

                        audio_float_array = self.phoneme_ids_to_audio(phoneme_ids, config)

                        # Post-processing
                        max_val = np.max(np.abs(audio_float_array))
                        if max_val >= 1e-8: audio_float_array = audio_float_array / max_val
                        if config.volume != 1.0: audio_float_array = audio_float_array * config.volume

                        audio_float_array = np.clip(audio_float_array, -1.0, 1.0).astype(np.float32)
                        audio_int16_bytes: bytes = (audio_float_array * 32767).astype(np.int16).tobytes()

                        # Chunking
                        CHUNK_SIZE = 8192
                        for i in range(0, len(audio_int16_bytes), CHUNK_SIZE):
                            chunk = audio_int16_bytes[i:i + CHUNK_SIZE]
                            # index_callback is not used for chunking in this TTS
                            audio_callback(chunk)

                    # Notify done speaking via index_callback(None)
                    index_callback(None)

                except Exception as e:
                    LOG.error(f"PatchedVoice: Error during synthesis: {e}")
                    # Ensure index_callback is called, even on error, to release the queue
                    index_callback(None)
                    return

        log.info("Phoonnx: TTSVoice and dependencies successfully imported (Wrapped).")

        return PatchedVoice, SynthesisConfig 

    except (ImportError, ModuleNotFoundError, AttributeError) as e:
        log.critical(f"FATAL ERROR: Failed to load Phoonnx or dependency. Check bundling: {e}", exc_info=True)
        return None, None 


class _VoiceLoaderThread(threading.Thread):
    """
    Asynchronously loads the TTSVoice instance and the WavePlayer.
    """

    def __init__(self, driver: 'SynthDriver', voice_id: str, model_path: str, config_path: str):
        super().__init__()
        self.driver = driver
        self.voice_id = voice_id
        self.model_path = model_path
        self.config_path = config_path
        self.daemon = True

    def run(self):
        log.info(f"Phoonnx ASYNC: Starting asynchronous loading of voice '{self.voice_id}'...")
        try:
            VoiceClass, SynthesisConfigClass = import_phoonnx()

            if VoiceClass is None or SynthesisConfigClass is None:
                raise PhoonnxException("TTS modules could not be loaded.")

            # Blocking load (uses PatchedVoice.load)
            tts_voice = VoiceClass.load(self.model_path, self.config_path)

            samplesPerSec = 22050
            if hasattr(tts_voice, 'sample_rate'):
                samplesPerSec = tts_voice.sample_rate

            player = WavePlayer(
                channels=1,
                samplesPerSec=samplesPerSec,
                bitsPerSample=16,
                purpose=AudioPurpose.SPEECH
            )

            self.driver.tts_voice = tts_voice
            self.driver._player = player
            self.driver._SynthesisConfig_class = SynthesisConfigClass

            log.info("Phoonnx ASYNC: Loading of TTSVoice and WavePlayer complete.")

        except Exception as e:
            log.error(f"Phoonnx ASYNC: Failed to load voice '{self.voice_id}': {e}", exc_info=True)
            self.driver.tts_voice = None
            self._player = None
            self.driver._SynthesisConfig_class = None 

        finally:
            self.driver._voice_loaded_event.set()


class _SynthQueueThread(threading.Thread):
    """
    Permanent worker thread that processes speech requests from the queue.
    """

    def __init__(self, driver: 'SynthDriver'):
        super().__init__()
        self.driver = driver
        self.daemon = True
        self.stop_event = threading.Event()
        self.cancel_synthesis_event = threading.Event()

    def run(self):
        log.info("Phoonnx SynthQueueThread: Starting permanent processing loop.")

        while not self.stop_event.is_set():
            try:
                request = self.driver._request_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if self.stop_event.is_set(): break

            try:
                text, config, index_callback, player_ref, tts_voice = request

                log.debug(f"Phoonnx QueueThread: Starting synthesis for: '{text[:20]}...'")

                self.cancel_synthesis_event.clear()
                stop_log_sent = False

                def thread_local_audio_callback(chunk: bytes) -> int:
                    """Sends audio and checks for cancellation requests."""
                    nonlocal stop_log_sent
                    if self.stop_event.is_set() or self.cancel_synthesis_event.is_set():
                        if not stop_log_sent:
                            log.debug("Phoonnx: Stop/Cancel event set. Synthesis actively stopped.")
                            stop_log_sent = True
                        return 1

                    try:
                        player_ref.feed(chunk)
                    except Exception:
                        return 1
                    return 0

                tts_voice.synthesize_to_callback(
                    text,
                    config=config,
                    audio_callback=thread_local_audio_callback,
                    index_callback=index_callback
                )

            except (PhoonnxException, Exception) as e:
                if not self.stop_event.is_set() and not self.cancel_synthesis_event.is_set():
                    log.error(f"Phoonnx Error (QueueThread): TTS synthesis failed: {e}", exc_info=True)

            finally:
                self.driver._request_queue.task_done()

        log.info("Phoonnx SynthQueueThread: Processing loop has stopped.")

    def cancel_synthesis(self):
        """Cancels the current synthesis in the thread."""
        self.cancel_synthesis_event.set()


class SynthDriver(BaseSynthDriver):
    """
    NVDA SynthDriver implementation for the Phoonnx TTS engine.
    """
    name = "phoonnx"
    description = _("Phoonnx TTS Driver")

    supportedCommands = frozenset([IndexCommand, PitchCommand, RateCommand, VolumeCommand, BreakCommand])
    supportedNotifications = frozenset([synthIndexReached, synthDoneSpeaking])

    supportedSettings = set(BaseSynthDriver.supportedSettings.fget(None)).union(
        [
            BaseSynthDriver.VoiceSetting(),
            BaseSynthDriver.RateSetting(),
            BaseSynthDriver.VolumeSetting(),
            BaseSynthDriver.PitchSetting(),
        ]
    )

    _AVAILABLE_VOICES_CONFIG: Dict[str, Dict[str, str]] = {}
    _availableVoicesCache: Optional[TOrderedDict[str, VoiceInfo]] = None
    _rate: int = 50
    _pitch: int = 50
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

        # LOADING OF CONFIGURATIONS AT INITIALIZATION
        SynthDriver._AVAILABLE_VOICES_CONFIG = load_voice_configs()

        if self.check():
            
            # --- START FIRST RUN LOGIC (SAFE IMPLEMENTATION) ---
            
            # Ensure the driver's configuration section exists
            if self.name not in config.conf["speech"]:
                config.conf["speech"][self.name] = {}
            
            # Retrieve the configuration section
            synth_config = config.conf["speech"][self.name]
            
            # Check if the first-run is already completed with the key 'initDone'
            if not synth_config.get(FIRST_RUN_KEY, False):
                log.info(f"Phoonnx: First run detected ({FIRST_RUN_KEY}=False). Initiating voice selection.")
                
                def deferred_voice_selection():
                    """Safely executes the voice selection dialog after the GUI is initialized."""
                    try:
                        import wx 
                        wx.CallAfter(voiceSelector.run_voice_selection)
                        log.info("Phoonnx: Deferred voice selection scheduled successfully.")
                    except Exception as e:
                        log.error(f"Phoonnx: Failed to schedule voice selection via wx.CallAfter: {e}")

                # Use threading.Timer to delay the wx.CallAfter call by a fraction of a second (0.5s).
                threading.Timer(0.5, deferred_voice_selection).start()
                    
                # Mark the first-run as complete and save the config
                synth_config[FIRST_RUN_KEY] = True
                config.conf.save()
                log.info(f"Phoonnx: First run logic completed and marked ({FIRST_RUN_KEY}=True).")
            # --- END FIRST RUN LOGIC ---

            self._worker_thread = _SynthQueueThread(driver=self)
            self._worker_thread.start()

        log.info("PHOONNX DEBUG: SynthDriver instance created.")

    @classmethod
    def check(cls) -> bool:
        if not cls._AVAILABLE_VOICES_CONFIG:
            cls._AVAILABLE_VOICES_CONFIG = load_voice_configs()

        if not cls._AVAILABLE_VOICES_CONFIG:
            log.warning(f"Phoonnx check failed: No valid voice configurations loaded.")
            return False

        return True

    def _getAvailableVoices(self) -> TOrderedDict[str, VoiceInfo]:
        if self._availableVoicesCache is None:
            self._availableVoicesCache = OrderedDict()
            for voice_id, config in SynthDriver._AVAILABLE_VOICES_CONFIG.items():
                if 'display_name' in config and 'language' in config:
                    self._availableVoicesCache[voice_id] = VoiceInfo(
                        voice_id, 
                        config['display_name'], 
                        language=config['language']
                    )
                else:
                    log.warning(f"Phoonnx: Voice config for '{voice_id}' is incomplete and skipped.")

        return self._availableVoicesCache

    def _get_voice(self) -> Optional[str]:
        available_voices = self.availableVoices
        
        # If the voice is not yet set, try to retrieve the saved preference.
        if self._voice_id is None: 
            
            # 1. Try to retrieve the user's saved preference via NVDA config
            saved_voice_id = None
            try:
                # config.getSynthConfig(self) retrieves the config object for this driver
                saved_voice_id = config.getSynthConfig(self).voice
                log.info(f"Phoonnx: Saved NVDA preferred voice: {saved_voice_id}")
            except Exception:
                log.warning("Phoonnx: Could not retrieve saved NVDA voice.")
            
            # 2. Determine the final voice ID
            if saved_voice_id and saved_voice_id in available_voices:
                new_voice_id = saved_voice_id
            elif available_voices:
                # Fall back to the hardcoded default or the first available
                default_voice_id = "dii_nl-NL" # Hardcoded fallback
                
                if default_voice_id in available_voices:
                    new_voice_id = default_voice_id
                else:
                    new_voice_id = list(available_voices.keys())[0]
                
                log.info(f"Phoonnx: No saved/valid voice found. Falling back to: {new_voice_id}")
            else:
                log.error("Phoonnx: No voices available in the configuration.")
                return None

            # 3. Load the voice
            self._voice_id = new_voice_id
            if self.check():
                self._load_tts_voice()
                
        # Make sure the voice is reloaded if the previous load failed (self.tts_voice is None)
        elif self.tts_voice is None:
             if self.check():
                self._load_tts_voice()
                
        return self._voice_id

    def _set_voice(self, value: str):
        if value not in self.availableVoices:
            log.warning(f"Phoonnx: Attempting to set invalid voice: {value}.")
            return
        
        # The BaseSynthDriver handles saving the value.
        # We only need to change and load the voice internally.
        if self._voice_id != value:
            self._voice_id = value
            self._voice_loaded_event.clear() 
            self._loader_thread = None 
            
            if self.check():
                self._load_tts_voice()

    def _load_tts_voice(self):
        voice_id = self._voice_id
        if voice_id is None:
            log.warning("Phoonnx: Attempted to load voice, but _voice_id is None.")
            self._voice_loaded_event.set() 
            return

        voice_config = SynthDriver._AVAILABLE_VOICES_CONFIG.get(voice_id)
        
        if not voice_config or 'model_file' not in voice_config or 'config_file' not in voice_config:
            log.error(f"Phoonnx: Voice configuration is invalid or incomplete for ID: {voice_id}. Cannot load.")
            self.tts_voice = None
            self._player = None
            self._SynthesisConfig_class = None
            self._voice_loaded_event.set()
            return

        model_file_relative = voice_config["model_file"]
        config_file_relative = voice_config["config_file"]
        
        # --- PATH CONSTRUCTION (with Normalization) ---
        model_path = os.path.normpath(os.path.join(DRIVER_DIR, model_file_relative))
        config_path = os.path.normpath(os.path.join(DRIVER_DIR, config_file_relative))

        log.info(f"Phoonnx: Trying to load voice '{voice_id}'. Model path: {model_path}")
        
        # Check if the files exist
        if not (os.path.exists(model_path) and os.path.exists(config_path)):
            log.error(f"Phoonnx: ERROR: Cannot find model or configuration file for '{voice_id}'.")
            
            # CRUCIAL LOGGING: Show the full paths that were NOT found
            if not os.path.exists(model_path):
                 log.error(f"Phoonnx: Error path model: {model_path} (File NOT FOUND)")
            if not os.path.exists(config_path):
                 log.error(f"Phoonnx: Error path config: {config_path} (File NOT FOUND)")

            self.tts_voice = None
            self._player = None
            self._SynthesisConfig_class = None
            self._voice_loaded_event.set()
            return

        # Start the loading process
        if not self._voice_loaded_event.is_set():
            log.info(f"Phoonnx: Starting asynchronous loading for voice '{self._voice_id}'.")
            
            self._loader_thread = _VoiceLoaderThread(
                driver=self,
                voice_id=voice_id,
                model_path=model_path, 
                config_path=config_path 
            )
            self._loader_thread.start()


    def _get_rate(self) -> int: return self._rate
    def _set_rate(self, value: int): self._rate = value

    def _get_volume(self) -> int: return self._volume
    def _set_volume(self, value: int): self._volume = value

    def _get_pitch(self) -> int: return self._pitch
    def _set_pitch(self, value: int): self._pitch = value

    def _get_language(self) -> Optional[str]:
        voice_info = self.availableVoices.get(self._get_voice())
        return voice_info.language if voice_info else None
    def _set_language(self, language): pass
    def _get_availableLanguages(self) -> Set[Optional[str]]:
        return {v.language for v in self.availableVoices.values()}

    def _onIndexReached(self, index: Optional[int]):
        if index is not None:
            synthIndexReached.notify(synth=self, index=index)
        else:
            synthDoneSpeaking.notify(synth=self)

    # --- Core Speech Control Functions ---
    def speak(self, speechSequence):
        
        if not self._voice_loaded_event.is_set():
            log.info("Phoonnx: Waiting for voice loading to complete (First speech or after voice change).")
            if not self._voice_loaded_event.wait(timeout=1):
                 log.error("Phoonnx: Voice loading timed out. Cannot speak.")
                 return
            log.info("Phoonnx: Voice successfully loaded after waiting.")

        if not self.tts_voice or not self._player or not self._SynthesisConfig_class:
            log.warning("Phoonnx: Cannot speak, TTS voice, WavePlayer, or Config Class not loaded.")
            return

        current_rate = self._get_rate()
        text = ""

        for item in speechSequence:
            if isinstance(item, str):
                text += item
            elif isinstance(item, RateCommand):
                current_rate = item.value
            elif isinstance(item, (PitchCommand, VolumeCommand, BreakCommand, IndexCommand)):
                pass

        nvda_rate = current_rate
        length_scale = 1.0 / (nvda_rate / 50.0)
        length_scale = max(0.2, min(2.0, length_scale))

        synthesis_config = self._SynthesisConfig_class( 
            length_scale=length_scale,
            noise_scale=0.667,
            noise_w_scale=0.8,
            enable_phonetic_spellings=True,
            add_diacritics=False
        )

        request = (text, synthesis_config, self._onIndexReached, self._player, self.tts_voice)
        self._request_queue.put(request)

    def cancel(self):
        if self._player:
            self._player.stop()

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.cancel_synthesis_event.set()

        while not self._request_queue.empty():
            try:
                self._request_queue.get(block=False)
                self._request_queue.task_done()
            except queue.Empty:
                break

    def pause(self, switch: bool):
        if self._player:
            self._player.pause(switch)

    def terminate(self):
        log.info("Phoonnx: Driver is terminating.")

        if self._player:
            self._player.close()

        if self._worker_thread and self._worker_thread.is_alive():
            log.info("Phoonnx: Shutting down QueueThread...")
            self._worker_thread.stop_event.set()
            self._worker_thread.join(timeout=1)

            if self._worker_thread.is_alive():
                log.warning("Phoonnx: QueueThread did not shut down within 1 second. Continuing.")

        self.tts_voice = None
        self._SynthesisConfig_class = None 

SynthDriver = SynthDriver
