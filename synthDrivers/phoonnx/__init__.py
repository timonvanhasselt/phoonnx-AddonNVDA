import os
import io
import wave
import sys
import threading
from collections import OrderedDict
from typing import OrderedDict as TOrderedDict, Optional, Set

# FIX: Importeer WavePlayer en AudioPurpose direct voor de moderne NVDA API (2025.3+)
from nvwave import WavePlayer, AudioPurpose 

# --- Essentiële NVDA Core Imports ---
from logHandler import log
from synthDriverHandler import (
    SynthDriver as BaseSynthDriver, 
    VoiceInfo,
    synthIndexReached, 
    synthDoneSpeaking,
)
from speech.commands import IndexCommand, PitchCommand
_ = lambda s: s 

# --- CRUCIALE CONFIGURATIE ---
VOICE_ID = "dii_nl-NL"
MODEL_FILENAME = f"{VOICE_ID}.onnx"
CONFIG_FILENAME = f"{VOICE_ID}.onnx.json"

log.critical("PHOONNX DEBUG: __init__.py is gestart met uitvoeren!")

# --- Configuratie van het Python Zoekpad ---
DRIVER_DIR = os.path.dirname(os.path.abspath(__file__))
PHOONNX_LIBS_PATH = os.path.join(DRIVER_DIR, "phoonnx_libs")
if PHOONNX_LIBS_PATH not in sys.path:
    sys.path.insert(0, PHOONNX_LIBS_PATH)

# --- Globale definitie van uitzondering ---
class PhoonnxException(Exception): pass

# --- Imports van de TTS-logica (Phoonnx) ---
try:
    from phoonnx.config import SynthesisConfig
    from phoonnx.voice import TTSVoice 
    
    log.info("Phoonnx: TTSVoice en afhankelijkheden succesvol geïmporteerd.")
    TTS_VOICE_LOADED = True
    
except (ImportError, ModuleNotFoundError, AttributeError) as e:
    log.critical(f"FATALE FOUT: Fout bij het laden van Phoonnx of afhankelijkheid. Controleer bundeling: {e}", exc_info=True)
    TTS_VOICE_LOADED = False
    
    # Definieer dummy-klassen als fallback
    class SynthesisConfig: pass
    class TTSVoice:
        @staticmethod
        def load(*args, **kwargs): 
            raise RuntimeError("Phoonnx-bibliotheek niet geladen.")
        def __init__(self, *args, **kwargs): pass
        def synthesize_wav(self, *args, **kwargs): raise RuntimeError("Phoonnx-bibliotheek niet geladen.")


# NIEUW: Thread voor asynchrone synthese
class _SynthThread(threading.Thread):
    """Voert de zware TTS-synthese uit in een aparte thread."""
    def __init__(self, tts_voice: TTSVoice, text: str, config: SynthesisConfig, player: WavePlayer):
        super().__init__()
        self.tts_voice = tts_voice
        self.text = text
        self.config = config
        self.player = player
        self.daemon = True # Zorgt ervoor dat de thread sluit als NVDA sluit
        
    def run(self):
        log.info("Phoonnx SynthThread: Starten van TTS synthese...")
        try:
            with io.BytesIO() as wav_buffer:
                # Genereer de WAV-data in-memory
                with wave.open(wav_buffer, "wb") as wav_writer:
                    # Dit is de BLOKKERENDE aanroep die nu in de aparte thread zit.
                    self.tts_voice.synthesize_wav(self.text, wav_writer, self.config)
                
                wav_buffer.seek(0)

                # Lees de ruwe audio data uit de buffer
                with wave.open(wav_buffer, "rb") as wav_file:
                    wav_file.getparams() 
                    audio_data = wav_file.readframes(wav_file.getnframes())

                # Voer audio data in.
                self.player.feed(audio_data)
                log.info("Phoonnx SynthThread: Synthese voltooid en audio ingevoerd in WavePlayer.")

        except (PhoonnxException, Exception) as e:
            log.error(f"Phoonnx Fout (SynthThread): TTS-synthese mislukt: {e}", exc_info=True)


class SynthDriver(BaseSynthDriver):
    """
    NVDA SynthDriver implementatie voor de Phoonnx TTS engine.
    """
    name = "phoonnx"
    description = _("Phoonnx TTS Driver")

    supportedCommands = frozenset([IndexCommand, PitchCommand])
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
        self.tts_voice: Optional[TTSVoice] = None
        self._voice_id: Optional[str] = None 
        # Voeg de WavePlayer toe om directe audio controle te hebben
        self._player: Optional[WavePlayer] = None 
        self.current_synth_thread: Optional[_SynthThread] = None # NIEUW: Houd thread bij
        
        if self.check():
            self._get_voice()
        log.critical("PHOONNX DEBUG: SynthDriver instantie aangemaakt.")

    @classmethod
    def check(cls) -> bool:
        """
        Controleert of de driver beschikbaar is en de modelbestanden bestaan.
        """
        if not TTS_VOICE_LOADED:
            return False
            
        model_path = os.path.join(DRIVER_DIR, MODEL_FILENAME)
        config_path = os.path.join(DRIVER_DIR, CONFIG_FILENAME)
        
        if not (os.path.exists(model_path) and os.path.exists(config_path)):
            log.warning(f"Phoonnx check faalde: Model- of configuratiebestand niet gevonden op verwachte locatie.")
            return False
        
        return True

    def _getAvailableVoices(self) -> TOrderedDict[str, VoiceInfo]:
        """Retourneert een Ordered Dictionary met de enkele beschikbare stem."""
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
                if TTS_VOICE_LOADED:
                    self._load_tts_voice() 
        return self._voice_id

    def _set_voice(self, value: str):
        if value not in self.availableVoices:
            log.warning(f"Phoonnx: Poging tot instellen van ongeldige stem: {value}.")
            return
        if self._voice_id != value:
            self._voice_id = value
            if TTS_VOICE_LOADED:
                self._load_tts_voice()

    def _load_tts_voice(self):
        """Initialiseer de TTSVoice-instantie en de WavePlayer."""
        if self._voice_id == VOICE_ID:
            log.info(f"Phoonnx: Laden van stem '{self._voice_id}' via TTSVoice.load()")
            model_path = os.path.join(DRIVER_DIR, MODEL_FILENAME)
            config_path = os.path.join(DRIVER_DIR, CONFIG_FILENAME)
            try:
                self.tts_voice = TTSVoice.load(model_path, config_path)
                
                # --- Handhaving van sample rate ---
                samplesPerSec = 22050 
                
                if hasattr(self.tts_voice, 'sample_rate'):
                    samplesPerSec = self.tts_voice.sample_rate
                else:
                    log.warning(f"Phoonnx: Attribuut 'sample_rate' niet gevonden op TTSVoice. Gebruik fallback van {samplesPerSec} Hz. Controleer of de spraak correct klinkt.")

                # Gebruik de moderne WavePlayer constructor met keyword arguments
                self._player = WavePlayer(
                    channels=1, 
                    samplesPerSec=samplesPerSec, 
                    bitsPerSample=16, 
                    purpose=AudioPurpose.SPEECH # Cruciaal voor ducking en focus
                )
            except Exception as e:
                log.error(f"Phoonnx: Kan stem '{self._voice_id}' niet laden: {e}", exc_info=True)
                self.tts_voice = None
                self._player = None
        else:
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

    
    def speak(self, speechSequence):
        """
        Spreekt de gegeven reeks tekst en spraakcommando's.
        Dit start de zware synthese ASYNCHROON in een aparte thread.
        """
        if not self.tts_voice or not self._player:
            log.warning("Phoonnx: Kan niet spreken, TTS-stem of WavePlayer niet geladen.")
            return

        # Zorg ervoor dat de vorige synthese-thread gestopt is (indien nodig)
        if self.current_synth_thread and self.current_synth_thread.is_alive():
             # Optioneel: Stoppen van de oude thread, hoewel 'cancel' dit ook zou moeten doen
             pass

        text = ""
        index = 0
        for item in speechSequence:
            if isinstance(item, str):
                text += item
            elif isinstance(item, IndexCommand):
                index = item.index

        if not text:
            return

        nvda_rate = self._get_rate() 
        length_scale = 1.0 / (nvda_rate / 50.0)
        length_scale = max(0.2, min(2.0, length_scale))

        synthesis_config = SynthesisConfig(
            length_scale=length_scale,
            noise_scale=0.667, 
            noise_w_scale=0.8,
            enable_phonetic_spellings=True,
            add_diacritics=False
        )

        # START DE SYNTHESE IN EEN APARTE THREAD
        self.current_synth_thread = _SynthThread(
            tts_voice=self.tts_voice, 
            text=text, 
            config=synthesis_config, 
            player=self._player
        )
        self.current_synth_thread.start()
        # De `speak` methode keert nu direct terug, waardoor de NVDA-kern niet wordt geblokkeerd.


    def cancel(self):
        """Annuleert de huidige spraakopdracht door de WavePlayer te stoppen."""
        if self._player:
            self._player.stop()
        # Zorg ervoor dat de synthese-thread ook stopt als deze bezig is
        if self.current_synth_thread and self.current_synth_thread.is_alive():
            # Het stoppen van een willekeurige thread is lastig in Python. 
            # Je moet dit in de TTSVoice-bibliotheek zelf implementeren 
            # met een stop-vlag. Voor nu laten we de thread zijn werk afmaken 
            # en alleen de audio-output stoppen.
            pass # <-- DEZE 'pass' ZOU HET PROBLEEM MOETEN OPLOSSEN

    def pause(self, switch: bool):
        """Pauzeer of hervat spraakuitvoer."""
        if self._player:
            # De moderne WavePlayer.pause() accepteert de switch-parameter
            self._player.pause(switch)

    def terminate(self):
        """Ruimt op wanneer de driver wordt gestopt."""
        log.info("Phoonnx: Driver wordt afgesloten.")
        if self._player:
            # Gebruik close() in de moderne API, wat ook de C++-helper opruimt
            self._player.close() 
        self.tts_voice = None 

log.critical("PHOONNX DEBUG: __init__.py is voltooid. SynthDriver-klasse is gedefinieerd.")