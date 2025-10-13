import os
import threading
from collections import OrderedDict
from typing import OrderedDict as TOrderedDict, Optional, Set
import queue 

# FIX: Import WavePlayer and AudioPurpose directly for the modern NVDA API (2025.3+)
from nvwave import WavePlayer, AudioPurpose 

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
VOICE_ID = "dii_nl-NL"
MODEL_FILENAME = f"{VOICE_ID}.onnx"
CONFIG_FILENAME = f"{VOICE_ID}.onnx.json"

log.debug("PHOONNX DEBUG: __init__.py has started execution.")

# --- Python Search Path Configuration ---
DRIVER_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Global Exception Definition ---
class PhoonnxException(Exception): pass


def import_phoonnx():
    from phoonnx.config import SynthesisConfig
    from phoonnx.voice import TTSVoice, LOG
    import numpy as np


    class PatchedVoice(TTSVoice):

        def synthesize_to_callback(self,
                                   text: str,
                                   audio_callback: callable,
                                   index_callback: callable,
                                   config: Optional[SynthesisConfig] = None,
                                   speaker_id: Optional[int] = None):
            """
            Synthetiseert tekst en streamt ruwe 16-bit audio naar de audio_callback.
            """
            if config is None:
                config = SynthesisConfig()

            LOG.debug("text=%s", text)

            if self.phonetic_spellings and config.enable_phonetic_spellings:
                text = self.phonetic_spellings.apply(text)

            if config.add_diacritics:
                text = self.phonemizer.add_diacritics(text, self.config.lang_code)
                LOG.debug("text+diacritics=%s", text)

            sentence_phonemes = self.phonemize(text)
            LOG.debug("phonemes=%s", sentence_phonemes)

            all_phoneme_ids_for_synthesis = [
                self.phonemes_to_ids(phonemes) for phonemes in sentence_phonemes if phonemes
            ]

            first_chunk = True
            sentence_silence = 0.0
            silence_int16_bytes = bytes(
                int(self.config.sample_rate * sentence_silence * 2)
            )

            for phoneme_ids in all_phoneme_ids_for_synthesis:
                if not phoneme_ids:
                    continue

                if not first_chunk:
                    audio_callback(silence_int16_bytes)
                first_chunk = False

                # Synthese (Blokkeert, maar in de thread van NVDA)
                audio_float_array = self.phoneme_ids_to_audio(phoneme_ids, config)

                # Post-processing
                if config.normalize_audio:
                    max_val = np.max(np.abs(audio_float_array))
                    if max_val < 1e-8:
                        audio_float_array = np.zeros_like(audio_float_array)
                    else:
                        audio_float_array = audio_float_array / max_val

                if config.volume != 1.0:
                    audio_float_array = audio_float_array * config.volume

                audio_float_array = np.clip(audio_float_array, -1.0, 1.0).astype(np.float32)

                # Converteer naar ruwe 16-bit bytes
                audio_int16_bytes: bytes = (audio_float_array * 32767).astype(np.int16).tobytes()

                # Chunking en verzenden naar de callback
                CHUNK_SIZE = 8192
                for i in range(0, len(audio_int16_bytes), CHUNK_SIZE):
                    chunk = audio_int16_bytes[i:i + CHUNK_SIZE]
                    audio_callback(chunk)

            # Einde van de synthese: signaleert NVDA dat het spreken klaar is
            index_callback(None)

    log.info("Phoonnx: TTSVoice and dependencies successfully imported.")
    return PatchedVoice

# =========================================================================
# ASYNCHRONOUS LOADING AND STREAMING LOGIC (REVISED WITH QUEUE)
# =========================================================================

class _VoiceLoaderThread(threading.Thread):
    """Asynchronously loads the TTSVoice instance and the WavePlayer."""
    
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
            # This is a blocking call in a separate thread
            tts_voice =  import_phoonnx().load(self.model_path, self.config_path)

            samplesPerSec = 22050
            if hasattr(tts_voice, 'sample_rate'):
                samplesPerSec = tts_voice.sample_rate
            
            # Create the WavePlayer
            player = WavePlayer(
                channels=1, 
                samplesPerSec=samplesPerSec, 
                bitsPerSample=16, 
                purpose=AudioPurpose.SPEECH 
            )

            # Change state in the main driver
            self.driver.tts_voice = tts_voice
            self.driver._player = player
            
            log.info("Phoonnx ASYNC: Loading of TTSVoice and WavePlayer complete.")

        except Exception as e:
            log.error(f"Phoonnx ASYNC: Failed to load voice '{self.voice_id}': {e}", exc_info=True)
            self.driver.tts_voice = None
            self.driver._player = None
            
        finally:
            # Set the event
            self.driver._voice_loaded_event.set()


class _SynthQueueThread(threading.Thread):
    """
    Permanent worker thread that processes speech requests from the queue.
    """
    
    def __init__(self, driver: 'SynthDriver'):
        super().__init__()
        self.driver = driver
        self.daemon = True 
        # Event to permanently shut down the thread
        self.stop_event = threading.Event() 
        # Event to actively stop the current synthesis
        self.cancel_synthesis_event = threading.Event() 
        
    def run(self):
        log.info("Phoonnx SynthQueueThread: Starting permanent processing loop.")
        
        while not self.stop_event.is_set():
            try:
                # Blocking get, but with a timeout to allow checking the stop_event
                request = self.driver._request_queue.get(timeout=0.1) 
            except queue.Empty:
                continue

            # Check again if the thread should not be terminated
            if self.stop_event.is_set():
                break
                
            try:
                # Unpack the request parameters
                text, config, index_callback, player_ref, tts_voice = request 
                
                log.debug(f"Phoonnx QueueThread: Starting synthesis for: '{text[:20]}...'")

                self.cancel_synthesis_event.clear() # Reset the cancellation flag
                stop_log_sent = False 
                
                # Local callback function
                def thread_local_audio_callback(chunk: bytes) -> int:
                    """Streams audio and checks for cancellation requests."""
                    
                    nonlocal stop_log_sent

                    # Check if the thread or the current synthesis should be cancelled
                    if self.stop_event.is_set() or self.cancel_synthesis_event.is_set():
                        if not stop_log_sent:
                            log.debug("Phoonnx: Stop/Cancel event set. Synthesis actively stopped.")
                            stop_log_sent = True 
                        return 1 # Non-zero return to tell TTSVoice to stop
                        
                    try:
                        player_ref.feed(chunk)
                    except Exception:
                        # Stop synthesis if the player fails/closes
                        return 1 

                    return 0 
                
                # Execute the blocking synthesis
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
                # Mark the task as done
                self.driver._request_queue.task_done()

        log.info("Phoonnx SynthQueueThread: Processing loop has stopped.")


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

    _availableVoicesCache: Optional[TOrderedDict[str, VoiceInfo]] = None
    _rate: int = 50
    _pitch: int = 50
    _volume: int = 100

    def __init__(self):
        super(SynthDriver, self).__init__()
        self.tts_voice: Optional['PatchedVoice'] = None
        self._voice_id: Optional[str] = None 
        self._player: Optional[WavePlayer] = None 
        
        # Permanent queue and worker thread
        self._request_queue: queue.Queue = queue.Queue()
        self._worker_thread: Optional[_SynthQueueThread] = None
        
        # Event and thread for asynchronous loading
        self._voice_loaded_event = threading.Event()
        self._loader_thread: Optional[_VoiceLoaderThread] = None 

        if self.check():
            # Call _get_voice to start asynchronous loading
            self._get_voice()
            
            # Start the permanent worker thread
            self._worker_thread = _SynthQueueThread(driver=self)
            self._worker_thread.start()
            
        log.info("PHOONNX DEBUG: SynthDriver instance created.")

    @classmethod
    def check(cls) -> bool:
        model_path = os.path.join(DRIVER_DIR, MODEL_FILENAME)
        config_path = os.path.join(DRIVER_DIR, CONFIG_FILENAME)
        if not (os.path.exists(model_path) and os.path.exists(config_path)):
            log.warning(f"Phoonnx check failed: Model or configuration file not found at expected location.")
            return False
        
        return True

    # --- Property/Setting Getters and Setters ---
    def _getAvailableVoices(self) -> TOrderedDict[str, VoiceInfo]:
        if self._availableVoicesCache is None:
            self._availableVoicesCache = OrderedDict()
            language = VOICE_ID.split('_')[-1] if '_' in VOICE_ID else None
            display_name = f"Phoonnx ({VOICE_ID.replace('_', ' ').upper()})"
            
            self._availableVoicesCache[VOICE_ID] = VoiceInfo(VOICE_ID, display_name, language=language)
                
        return self._availableVoicesCache

    def _get_voice(self) -> Optional[str]:
        if self._voice_id is None:
            available_voices = self.availableVoices
            if available_voices:
                self._voice_id = list(available_voices.keys())[0]
                self._load_tts_voice()
        return self._voice_id

    def _set_voice(self, value: str):
        if value not in self.availableVoices:
            log.warning(f"Phoonnx: Attempting to set invalid voice: {value}.")
            return
        if self._voice_id != value:
            self._voice_id = value
            self._load_tts_voice()

    def _load_tts_voice(self):
        """Starts the asynchronous thread to initialize the TTSVoice instance and the WavePlayer."""
        if self._voice_id == VOICE_ID and not self._voice_loaded_event.is_set() and self._loader_thread is None:
            log.info(f"Phoonnx: Starting asynchronous loading for voice '{self._voice_id}'.")
            
            model_path = os.path.join(DRIVER_DIR, MODEL_FILENAME)
            config_path = os.path.join(DRIVER_DIR, CONFIG_FILENAME)
            
            self._loader_thread = _VoiceLoaderThread(
                driver=self, 
                voice_id=self._voice_id, 
                model_path=model_path, 
                config_path=config_path
            )
            self._loader_thread.start()
        
        elif self._voice_id != VOICE_ID:
            self.tts_voice = None
            self._player = None
            
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
        """This callback is called from the permanent worker thread."""
        if index is not None:
            synthIndexReached.notify(synth=self, index=index)
        else:
            synthDoneSpeaking.notify(synth=self)
    
    # --- Core Speech Control Functions ---
    def speak(self, speechSequence):
        """
        Adds a speech request to the queue.
        """
        # The line `self.cancel()` was removed here to allow subsequent speech requests 
        # (e.g., from NVDA's "Say All" command) to be correctly queued instead of cancelling 
        # the currently running speech.
        
        # Wait until the model is loaded, if necessary.
        if not self._voice_loaded_event.is_set():
            log.info("Phoonnx: Waiting for voice loading to complete (First speech).")
            self._voice_loaded_event.wait()
            log.info("Phoonnx: Voice successfully loaded after waiting.")

        if not self.tts_voice or not self._player:
            log.warning("Phoonnx: Cannot speak, TTS voice or WavePlayer not loaded.")
            return

        # STEP 1: TEXT PREPARATION & COMMAND HANDLING
        current_rate = self._get_rate() 
        text = ""
        
        for item in speechSequence:
            if isinstance(item, str):
                text += item
            elif isinstance(item, RateCommand):
                current_rate = item.value 
                log.debug(f"Phoonnx: RateCommand received (Value: {item.value}), used for calculation.")
            elif isinstance(item, (PitchCommand, VolumeCommand, BreakCommand, IndexCommand)):
                pass 

        if not text:
            return

        # STEP 2: CONFIGURATION (calculate speed)
        nvda_rate = current_rate
        length_scale = 1.0 / (nvda_rate / 50.0)
        length_scale = max(0.2, min(2.0, length_scale))

        from phoonnx.config import SynthesisConfig
        synthesis_config = SynthesisConfig(
            length_scale=length_scale,
            noise_scale=0.667, 
            noise_w_scale=0.8,
            enable_phonetic_spellings=True,
            add_diacritics=False
        )

        # STEP 3: PLACE THE REQUEST IN THE QUEUE
        request = (text, synthesis_config, self._onIndexReached, self._player, self.tts_voice)
        self._request_queue.put(request)


    def cancel(self):
        """Cancels the current speech command and clears the queue."""
        
        # 1. Stop the audio output immediately
        if self._player:
            self._player.stop() 
        
        # 2. Request the worker thread to cancel the current synthesis
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.cancel_synthesis_event.set()
            
        # 3. Clear all remaining tasks from the queue
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
        """
        Ensures the shutdown of the worker thread, WavePlayer, and driver.
        """
        log.info("Phoonnx: Driver is terminating.")
        
        # 1. Close the WavePlayer.
        if self._player:
            self._player.close() 
            
        # 2. Stop the permanent worker thread and wait for shutdown.
        if self._worker_thread and self._worker_thread.is_alive():
            log.info("Phoonnx: Shutting down QueueThread...")
            # Signal the thread to stop the loop
            self._worker_thread.stop_event.set()
            # Wait max 1 second for a clean shutdown
            self._worker_thread.join(timeout=1) 
            if self._worker_thread.is_alive():
                log.warning("Phoonnx: QueueThread did not shut down within 1 second. Continuing.")
                
        self.tts_voice = None 

log.debug("PHOONNX DEBUG: __init__.py complete. SynthDriver class is defined.")
