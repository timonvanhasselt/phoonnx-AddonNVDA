import os
import sys
import threading
from collections import OrderedDict
from typing import OrderedDict as TOrderedDict, Optional, Set, Callable
import queue 
import time


# Importeer phoonnx types om type hinting in de driver te vereenvoudigen
try:
    from phoonnx.config import SynthesisConfig 
    from phoonnx.voice import TTSVoice, LOG 
except (ImportError, ModuleNotFoundError):
    # Dit is prima, de werkelijke import gebeurt lazy in import_phoonnx()
    class SynthesisConfig: pass
    class TTSVoice: pass
    class LOG: 
        @staticmethod
        def debug(*args, **kwargs): pass
        @staticmethod
        def error(*args, **kwargs): pass
        @staticmethod
        def info(*args, **kwargs): pass


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

# --- Python Search Path Configuration (BEHOUDEN voor gebundelde libs) ---
DRIVER_DIR = os.path.dirname(os.path.abspath(__file__))
PHOONNX_LIBS_PATH = os.path.join(DRIVER_DIR, "phoonnx_libs")
if PHOONNX_LIBS_PATH not in sys.path:
    sys.path.insert(0, PHOONNX_LIBS_PATH)

# --- Global Exception Definition ---
class PhoonnxException(Exception): pass


def import_phoonnx():
    """
    Importeert Phoonnx en retourneert de gepatchte TTSVoice klasse EN SynthesisConfig.
    """
    try:
        from phoonnx.config import SynthesisConfig
        from phoonnx.voice import TTSVoice, LOG 
        import numpy as np # <-- Lazy import van numpy 
        
        class PatchedVoice(TTSVoice):
            """
            Patched TTSVoice klasse met de volledige synthese- en chunking logica.
            """
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
                            # index_callback wordt niet gebruikt voor chunking in deze TTS
                            audio_callback(chunk)

                    # Notify done speaking via index_callback(None)
                    index_callback(None) 
                    
                except Exception as e:
                    LOG.error(f"PatchedVoice: Fout tijdens synthese: {e}")
                    # Zorg ervoor dat index_callback wordt aangeroepen, zelfs bij een fout, om de wachtrij vrij te geven
                    index_callback(None) 
                    return
            
        log.info("Phoonnx: TTSVoice and dependencies successfully imported.")
        
        return PatchedVoice, SynthesisConfig # <--- RETOURNEER NU BEIDE

    except (ImportError, ModuleNotFoundError, AttributeError) as e:
        log.critical(f"FATAL ERROR: Failed to load Phoonnx or dependency. Check bundling: {e}", exc_info=True)
        return None, None # <--- Retourneer twee None's bij fout


# =========================================================================
# ASYNCHRONOUS LOADING AND STREAMING LOGIC
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
            # Vang nu beide geretourneerde waardes op
            VoiceClass, SynthesisConfigClass = import_phoonnx()
            
            if VoiceClass is None or SynthesisConfigClass is None:
                raise PhoonnxException("TTS modules konden niet geladen worden.")

            # Blocking load (gebruikt PatchedVoice.load)
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
            # SLAG DE GELADEN CONFIG KLASSE OP IN DE DRIVER
            self.driver._SynthesisConfig_class = SynthesisConfigClass
            
            log.info("Phoonnx ASYNC: Loading of TTSVoice and WavePlayer complete.")

        except Exception as e:
            log.error(f"Phoonnx ASYNC: Failed to load voice '{self.voice_id}': {e}", exc_info=True)
            self.driver.tts_voice = None
            self.driver._player = None
            self.driver._SynthesisConfig_class = None # Zorg ervoor dat deze ook None is
            
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
                # request is: (text, config, index_callback, player_ref, tts_voice) - 5 elementen
                request = self.driver._request_queue.get(timeout=0.1) 
            except queue.Empty:
                continue

            if self.stop_event.is_set(): break
                
            try:
                # Ontpak 5 elementen (Structuur van refactor)
                text, config, index_callback, player_ref, tts_voice = request 
                
                self.cancel_synthesis_event.clear() 
                stop_log_sent = False 
                
                def thread_local_audio_callback(chunk: bytes) -> int:
                    """Stuurt audio en controleert op annuleringsverzoeken."""
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

                # CRUCIALE FIX BEHOUDEN: Wacht tot de WavePlayer klaar is met afspelen.
                # Dit is essentieel voor NVDA.
                # player_ref.idle() # <--- DEZE REGEL IS VERWIJDERD OM DE RESPONSIVITEIT TE VERBETEREN
                
            except (PhoonnxException, Exception) as e:
                if not self.stop_event.is_set() and not self.cancel_synthesis_event.is_set():
                    log.error(f"Phoonnx Error (QueueThread): TTS synthesis failed: {e}", exc_info=True)
            
            finally:
                self.driver._request_queue.task_done()

        log.info("Phoonnx SynthQueueThread: Processing loop has stopped.")
        
    def cancel_synthesis(self):
        """Annuleert de huidige synthese in de thread."""
        self.cancel_synthesis_event.set()


class SynthDriver(BaseSynthDriver):
    """
    NVDA SynthDriver implementation for the Phoonnx TTS engine.
    (Bijgewerkt met schone structuur en index callback logica van refactor)
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
        # Type hinting veranderd naar de basisklasse
        self.tts_voice: Optional[TTSVoice] = None 
        self._voice_id: Optional[str] = None 
        self._player: Optional[WavePlayer] = None 
        self._SynthesisConfig_class: Optional[type] = None # <--- NIEUW ATTRIBUUT VOOR DE GELADEN CONFIG KLASSE
        
        self._request_queue: queue.Queue = queue.Queue()
        self._worker_thread: Optional[_SynthQueueThread] = None
        
        self._voice_loaded_event = threading.Event()
        self._loader_thread: Optional[_VoiceLoaderThread] = None 

        if self.check():
            self._get_voice() 
            
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

    def _getAvailableVoices(self) -> TOrderedDict[str, VoiceInfo]:
        """
        Gebruikt de correcte aanroep voor VoiceInfo. 
        (Bijgewerkt met generiekere display naam van refactor)
        """
        if self._availableVoicesCache is None:
            self._availableVoicesCache = OrderedDict()
            language = VOICE_ID.split('_')[-1] if '_' in VOICE_ID else None
            # Gebruik de generiekere display naam van de refactor
            display_name = f"Phoonnx ({VOICE_ID.replace('_', ' ').upper()})"
            
            self._availableVoicesCache[VOICE_ID] = VoiceInfo(VOICE_ID, display_name, language=language)
                
        return self._availableVoicesCache

    def _get_voice(self) -> Optional[str]:
        if self._voice_id is None:
            available_voices = self.availableVoices
            if available_voices:
                self._voice_id = list(available_voices.keys())[0]
                if self.check():
                    self._load_tts_voice() 
        return self._voice_id

    def _set_voice(self, value: str):
        if value not in self.availableVoices:
            log.warning(f"Phoonnx: Attempting to set invalid voice: {value}.")
            return
        if self._voice_id != value:
            self._voice_id = value
            if self.check():
                self._load_tts_voice()

    def _load_tts_voice(self):
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
            self._SynthesisConfig_class = None # Wist ook de config klasse bij stemwisseling
            
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
        """
        Callback van de worker thread om NVDA te notificeren.
        """
        if index is not None:
            synthIndexReached.notify(synth=self, index=index) 
        else:
            # Dit notificeert synthDoneSpeaking.
            synthDoneSpeaking.notify(synth=self) 
    
    # --- Core Speech Control Functions ---
    def speak(self, speechSequence):
        """
        Voegt een spraakverzoek toe aan de queue.
        """
        
        # Wacht tot de TTS-stem volledig is geladen (Overgenomen van refactor)
        if not self._voice_loaded_event.is_set():
            log.info("Phoonnx: Waiting for voice loading to complete (First speech).")
            self._voice_loaded_event.wait()
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

        if not text: return

        # CONFIGURATION (calculate speed)
        nvda_rate = current_rate
        length_scale = 1.0 / (nvda_rate / 50.0)
        length_scale = max(0.2, min(2.0, length_scale))

        # Gebruik de opgeslagen SynthesisConfig klasse
        synthesis_config = self._SynthesisConfig_class( # <--- CORRECTIE
            length_scale=length_scale,
            noise_scale=0.667, 
            noise_w_scale=0.8,
            enable_phonetic_spellings=True,
            add_diacritics=False
        )
        
        # PLAATS HET VERZOEK IN DE QUEUE: (text, config, index_callback, player_ref, tts_voice)
        request = (text, synthesis_config, self._onIndexReached, self._player, self.tts_voice)
        self._request_queue.put(request)

    def cancel(self):
        # Behoud de logica, identiek aan refactor
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
        # Behoud de logica, identiek aan refactor
        log.info("Phoonnx: Driver is terminating.")
        
        if self._player:
            self._player.close() 
            
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.stop_event.set()
            self._worker_thread.join(timeout=1) 
                
        self.tts_voice = None 
        self._SynthesisConfig_class = None # Wist ook de config klasse bij beëindiging

log.debug("PHOONNX DEBUG: __init__.py complete. SynthDriver class is defined.")

SynthDriver = SynthDriver
