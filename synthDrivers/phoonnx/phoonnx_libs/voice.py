import json
import os.path
import re
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Union, Dict, List

import numpy as np
import onnxruntime
from langcodes import closest_match

# Aannames over de interne structuur:
from phoonnx.config import PhonemeType, VoiceConfig, SynthesisConfig, get_phonemizer
from phoonnx.phoneme_ids import phonemes_to_ids, BlankBetween
from phoonnx.phonemizers import Phonemizer
from phoonnx.phonemizers.base import PhonemizedChunks

_PHONEME_BLOCK_PATTERN = re.compile(r"(\[\[.*?\]\])")

try:
    from ovos_utils.log import LOG
except ImportError:
    import logging

    LOG = logging.getLogger(__name__)
    LOG.setLevel("DEBUG")


@dataclass
class PhoneticSpellings:
    replacements: Dict[str, str] = field(default_factory=dict)

    @staticmethod
    def from_lang(lang: str, locale_path: str = f"{os.path.dirname(__file__)}/locale"):
        langs = os.listdir(locale_path)
        lang2, distance = closest_match(lang, langs)
        if distance <= 10:
            spellings_file = f"{locale_path}/{lang2}/phonetic_spellings.txt"
            return PhoneticSpellings.from_path(spellings_file)
        raise FileNotFoundError(f"Spellings file for '{lang}' not found")

    @staticmethod
    def from_path(spellings_file: str):
        replacements = {}
        with open(spellings_file, "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
            for l in lines:
                if ":" in l:
                    word, spelling = l.split(":", 1)
                    replacements[word.strip()] = spelling.strip()
        return PhoneticSpellings(replacements)

    def apply(self, text: str) -> str:
        for k, v in self.replacements.items():
            pattern = r'\b' + re.escape(k) + r'\b'
            text = re.sub(pattern, v, text, flags=re.IGNORECASE)
        return text


@dataclass
class AudioChunk:
    """Chunk of raw audio. (Wordt niet meer gebruikt door de driver, maar is noodzakelijk voor de interne synthesize generator)."""

    sample_rate: int
    sample_width: int
    sample_channels: int
    audio_float_array: np.ndarray

    _audio_int16_array: Optional[np.ndarray] = None
    _audio_int16_bytes: Optional[bytes] = None
    _MAX_WAV_VALUE: float = 32767.0

    @property
    def audio_int16_array(self) -> np.ndarray:
        if self._audio_int16_array is None:
            self._audio_int16_array = np.clip(
                self.audio_float_array * self._MAX_WAV_VALUE, -self._MAX_WAV_VALUE, self._MAX_WAV_VALUE
            ).astype(np.int16)

        return self._audio_int16_array

    @property
    def audio_int16_bytes(self) -> bytes:
        return self.audio_int16_array.tobytes()


def _get_onnx_session(model_path: str, use_cuda: bool = False):
    """Initialiseert de ONNX Runtime sessie."""
    providers = ["CPUExecutionProvider"]
    if use_cuda:
        providers.insert(0, "CUDAExecutionProvider")

    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    return onnxruntime.InferenceSession(
        model_path, 
        sess_options=sess_options, 
        providers=providers
    )


@dataclass
class TTSVoice:
    session: onnxruntime.InferenceSession

    config: VoiceConfig

    phonetic_spellings: Optional[PhoneticSpellings] = None

    phonemizer: Optional[Phonemizer] = None

    def __post_init__(self):
        try:
            self.phonetic_spellings = PhoneticSpellings.from_lang(self.config.lang_code)
        except FileNotFoundError:
            pass
        if self.phonemizer is None:
            # Controleer de phonemizer-instellingen uit config
            self.phonemizer = get_phonemizer(
                self.config.phoneme_type,
                self.config.alphabet,
                self.config.phonemizer_model
            )
        
        # OPLOSSING VOOR ATTRIBUTE ERROR:
        # De audio sample rate is waarschijnlijk direct op het hoofdconfiguratieobject:
        self.sample_rate = self.config.sample_rate # <--- Gecorrigeerde regel
        # Indien de oude structuur: self.sample_rate = self.config.audio.sample_rate

    @staticmethod
    def load(
            model_path: Union[str, Path],
            config_path: Optional[Union[str, Path]] = None,
            phonemes_txt: Optional[str] = None,
            phoneme_map: Optional[str] = None,
            lang_code: Optional[str] = None,
            phoneme_type_str: Optional[str] = None,
            use_cuda: bool = False,
            **kwargs # Toevoeging om ongebruikte kwargs op te vangen
    ) -> "TTSVoice":
        
        if config_path is None:
            config_path = f"{model_path}.json"
            LOG.debug("Guessing voice config path: %s", config_path)

        with open(config_path, "r", encoding="utf-8") as config_file:
            config_dict = json.load(config_file)

        providers: list[Union[str, tuple[str, dict[str, Any]]]]
        if use_cuda:
            providers = [
                (
                    "CUDAExecutionProvider",
                    {"cudnn_conv_algo_search": "HEURISTIC"},
                )
            ]
            LOG.debug("Using CUDA")
        else:
            providers = ["CPUExecutionProvider"]

        config = VoiceConfig.from_dict(config_dict,
                                         phonemes_txt=phonemes_txt,
                                         lang_code=lang_code,
                                         phoneme_type_str=phoneme_type_str)
        
        # Laatste instellingen uit kwargs overschrijven
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)


        return TTSVoice(
            config=config,
            session=_get_onnx_session(str(model_path), use_cuda)
        )

    def phonemize(self, text: str) -> PhonemizedChunks:
        """
        Text to phonemes grouped by sentence.
        """
        phonemes: list[list[str]] = []

        text_parts = _PHONEME_BLOCK_PATTERN.split(text)

        for i, text_part in enumerate(text_parts):
            if text_part.startswith("[["):
                # Phonemes
                if not phonemes:
                    phonemes.append([])

                if (i > 0) and (text_parts[i - 1].endswith(" ")):
                    phonemes[-1].append(" ")

                phonemes[-1].extend(list(text_part[2:-2].strip()))

                if (i < (len(text_parts)) - 1) and (text_parts[i + 1].startswith(" ")):
                    phonemes[-1].append(" ")

                continue

            # Phonemization
            new_phonemes = self.phonemizer.phonemize(
                text_part, self.config.lang_code
            )
            phonemes.extend(new_phonemes)


        if phonemes and (not phonemes[-1]):
            phonemes.pop()

        return phonemes

    def phonemes_to_ids(self, phonemes: list[str]) -> list[int]:
        """
        Phonemes to ids.
        """
        if self.config.phoneme_id_map is None:
            raise ValueError("self.config.phoneme_id_map is None")
        return phonemes_to_ids(phonemes, self.config.phoneme_id_map,
                               blank_token=self.config.blank_token,
                               bos_token=self.config.bos_token,
                               eos_token=self.config.eos_token,
                               word_sep_token=self.config.word_sep_token,
                               include_whitespace=self.config.include_whitespace,
                               blank_at_start=self.config.blank_at_start,
                               blank_at_end=self.config.blank_at_end,
                               blank_between=BlankBetween.TOKENS_AND_WORDS,
                               )
                               
    def phoneme_ids_to_audio(
            self, phoneme_ids: list[int], syn_config: Optional[SynthesisConfig] = None
    ) -> np.ndarray:
        """
        Synthesize raw audio from phoneme ids (float32).
        """
        if syn_config is None:
            syn_config = SynthesisConfig()

        langid = syn_config.lang_id or 0
        speaker_id = syn_config.speaker_id or 0
        length_scale = syn_config.length_scale
        noise_scale = syn_config.noise_scale
        noise_w_scale = syn_config.noise_w_scale

        expected_args = [model_input.name for model_input in self.session.get_inputs()]

        phoneme_ids_array = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
        phoneme_ids_lengths = np.array([phoneme_ids_array.shape[1]], dtype=np.int64)
        args = {
            "input": phoneme_ids_array,
            "input_lengths": phoneme_ids_lengths
        }

        if length_scale is None:
            length_scale = self.config.length_scale
        if noise_scale is None:
            noise_scale = self.config.noise_scale
        if noise_w_scale is None:
            noise_w_scale = self.config.noise_w_scale
        if "scales" in expected_args:
            args["scales"] = np.array(
                [noise_scale, length_scale, noise_w_scale],
                dtype=np.float32,
            )

        if "langid" in expected_args:
            args["langid"] = np.array([langid], dtype=np.int64)
        if "sid" in expected_args:
            args["sid"] = np.array([speaker_id], dtype=np.int64)


        args = {k: v for k, v in args.items() if k in expected_args}
        audio = self.session.run(
            None,
            args,
        )[0].squeeze()

        return audio

    
    # =====================================================================
    # STREAMING IMPLEMENTATIE (synthesize_to_callback)
    # =====================================================================

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


    # =====================================================================
    # synthesize EN synthesize_wav (Behouden voor compatibiliteit/intern gebruik)
    # =====================================================================
    
    def synthesize(
            self,
            text: str,
            syn_config: Optional[SynthesisConfig] = None,
    ) -> Iterable[AudioChunk]:
        """Synthesize one audio chunk per sentence from text (generator)."""
        if syn_config is None:
            syn_config = SynthesisConfig()

        LOG.debug("text=%s", text)

        if self.phonetic_spellings and syn_config.enable_phonetic_spellings:
            text = self.phonetic_spellings.apply(text)

        if syn_config.add_diacritics:
            text = self.phonemizer.add_diacritics(text, self.config.lang_code)
            LOG.debug("text+diacritics=%s", text)

        sentence_phonemes = self.phonemize(text)
        LOG.debug("phonemes=%s", sentence_phonemes)
        all_phoneme_ids_for_synthesis = [
            self.phonemes_to_ids(phonemes) for phonemes in sentence_phonemes if phonemes
        ]

        for phoneme_ids in all_phoneme_ids_for_synthesis:
            if not phoneme_ids:
                continue

            audio = self.phoneme_ids_to_audio(phoneme_ids, syn_config)

            if syn_config.normalize_audio:
                max_val = np.max(np.abs(audio))
                if max_val < 1e-8:
                    audio = np.zeros_like(audio)
                else:
                    audio = audio / max_val

            if syn_config.volume != 1.0:
                audio = audio * syn_config.volume

            audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

            yield AudioChunk(
                sample_rate=self.config.sample_rate,
                sample_width=2,
                sample_channels=1,
                audio_float_array=audio,
            )

    def synthesize_wav(
            self,
            text: str,
            wav_file: wave.Wave_write,
            syn_config: Optional[SynthesisConfig] = None,
            set_wav_format: bool = True,
    ) -> None:
        """Synthesize and write WAV audio from text (Gebruikt nu streaming intern)."""
        
        # 1. WAV-header configureren
        if set_wav_format:
            # GEBRUIK HIER OOK DE GECORRIGEERDE sample_rate
            wav_file.setframerate(self.config.sample_rate) 
            wav_file.setsampwidth(2)
            wav_file.setnchannels(1)

        # 2. Hulpfuncties voor callbacks
        audio_buffer = []

        def local_audio_callback(chunk: bytes):
            audio_buffer.append(chunk)

        def dummy_index_callback(index: Optional[int]):
            pass

        # 3. Roep de streaming methode aan
        self.synthesize_to_callback(
            text=text,
            audio_callback=local_audio_callback,
            index_callback=dummy_index_callback,
            config=syn_config,
            speaker_id=syn_config.speaker_id if syn_config else None
        )

        # 4. Schrijf de verzamelde audio naar het WAV-bestand
        wav_file.writeframes(b"".join(audio_buffer))

if __name__ == "__main__":
    from phoonnx.phonemizers.gl import CotoviaPhonemizer
    from phoonnx.phonemizers.he import PhonikudPhonemizer
    from phoonnx.phonemizers.mul import (EspeakPhonemizer, EpitranPhonemizer, GruutPhonemizer, ByT5Phonemizer)

    syn_config = SynthesisConfig(enable_phonetic_spellings=True)

    # test hebrew piper
    model = "/home/miro/PycharmProjects/phoonnx_tts/phonikud/model.onnx"
    config = "/home/miro/PycharmProjects/phoonnx_tts/phonikud/model.config.json"

    voice = TTSVoice.load(model_path=model, config_path=config, use_cuda=False)

    print("\n################")
    # hebrew phonemes (raw input model)
    pho = PhonikudPhonemizer(diacritics=True)
    sentence = "הכוח לשנות מתחיל ברגע שבו אתה מאמין שזה אפשרי!"
    sentence = pho.phonemize_string(sentence, "he")

    print("## piper hebrew (raw)")
    print("-", voice.config.phoneme_type)
    slug = f"piper_{voice.config.phoneme_type.value}_{voice.config.lang_code}"
    with wave.open(f"{slug}.wav", "wb") as wav_file:
        voice.synthesize_wav(sentence, wav_file, syn_config)

    print("\n################")
    sentence = "הכוח לשנות מתחיל ברגע שבו אתה מאמין שזה אפשרי!"
    voice.config.phoneme_type = PhonemeType.PHONIKUD
    voice.phonemizer = pho

    print("## piper hebrew (phonikud)")
    print("-", voice.config.phoneme_type)
    slug = f"piper_{voice.config.phoneme_type.value}_{voice.config.lang_code}"
    with wave.open(f"{slug}.wav", "wb") as wav_file:
        voice.synthesize_wav(sentence, wav_file, syn_config)

    exit()
    # test piper
    model = "/home/miro/PycharmProjects/phoonnx_tts/miro_en-GB.onnx"
    config = "/home/miro/PycharmProjects/phoonnx_tts/piper_espeak.json"

    voice = TTSVoice.load(model_path=model, config_path=config, use_cuda=False)
    byt5_phonemizer = ByT5Phonemizer()
    gruut_phonemizer = GruutPhonemizer()
    espeak_phonemizer = EspeakPhonemizer()
    epitran_phonemizer = EpitranPhonemizer()
    cotovia_phonemizer = CotoviaPhonemizer()

    sentence = "A rainbow is a meteorological phenomenon that is caused by reflection, refraction and dispersion of light in water droplets resulting in a spectrum of light appearing in the sky. It takes the form of a multi-colored circular arc. Rainbows caused by sunlight always appear in the section of sky directly opposite the Sun."

    print("\n################")
    print("## piper")
    for phonemizer_type, phonemizer in [
        (PhonemeType.ESPEAK, espeak_phonemizer),
        (PhonemeType.BYT5, byt5_phonemizer),
        (PhonemeType.GRUUT, gruut_phonemizer),
        (PhonemeType.EPITRAN, epitran_phonemizer)
    ]:
        voice.config.phoneme_type = phonemizer_type
        voice.phonemizer = phonemizer
        print("-", phonemizer_type)

        slug = f"piper_{phonemizer_type.value}_{voice.config.lang_code}"
        with wave.open(f"{slug}.wav", "wb") as wav_file:
            voice.synthesize_wav(sentence, wav_file, syn_config)

    print("\n################")
    print("## mimic3")
    model = "/home/miro/PycharmProjects/phoonnx_tts/mimic3_ap/generator.onnx"
    config = "/home/miro/PycharmProjects/phoonnx_tts/mimic3_ap/config.json"
    phonemes_txt = "/home/miro/PycharmProjects/phoonnx_tts/mimic3_ap/phonemes.txt"
    phoneme_map = "/home/miro/PycharmProjects/phoonnx_tts/mimic3_ap/phoneme_map.txt"

    voice = TTSVoice.load(model_path=model, config_path=config,
                          phonemes_txt=phonemes_txt, phoneme_map=phoneme_map,
                          use_cuda=False)
    for phonemizer_type, phonemizer in [
        (PhonemeType.ESPEAK, espeak_phonemizer),
        (PhonemeType.BYT5, byt5_phonemizer),
        (PhonemeType.GRUUT, gruut_phonemizer),
        (PhonemeType.EPITRAN, epitran_phonemizer)
    ]:
        voice.config.phoneme_type = phonemizer_type
        voice.phonemizer = phonemizer
        print("-", phonemizer_type)
        slug = f"mimic3_{voice.config.phoneme_type.value}_{voice.config.lang_code}"
        with wave.open(f"{slug}.wav", "wb") as wav_file:
            voice.synthesize_wav(sentence, wav_file, syn_config)

    # Test grapheme model directly
    print("\n################")
    print("## coqui vits")
    model = "/home/miro/PycharmProjects/phoonnx_tts/celtia_vits/model.onnx"
    config = "/home/miro/PycharmProjects/phoonnx_tts/celtia_vits/config.json"

    sentence = "Este é un sistema de conversión de texto a voz en lingua galega baseado en redes neuronais artificiais. Ten en conta que as funcionalidades incluídas nesta páxina ofrécense unicamente con fins de demostración. Se tes algún comentario, suxestión ou detectas algún problema durante a demostración, ponte en contacto connosco."

    voice = TTSVoice.load(model_path=model, config_path=config,
                          use_cuda=False, lang_code="gl-ES")
    print("-", voice.config.phoneme_type)
    print(voice.config)
    phones = voice.phonemize(sentence)
    print(phones)
    print(voice.phonemes_to_ids(phones[0]))

    slug = f"vits_{voice.config.phoneme_type.value}_{voice.config.lang_code}"
    with wave.open(f"{slug}.wav", "wb") as wav_file:
        voice.synthesize_wav(sentence, wav_file, syn_config)

    # Test cotovia phonemizer
    print("\n################")
    print("## cotovia coqui vits")
    model = "/home/miro/PycharmProjects/phoonnx_tts/sabela_cotovia/model.onnx"
    config = "/home/miro/PycharmProjects/phoonnx_tts/sabela_cotovia/config.json"

    sentence = "Este é un sistema de conversión de texto a voz en lingua galega baseado en redes neuronais artificiais. Ten en conta que as funcionalidades incluídas nesta páxina ofrécense unicamente con fins de demostración. Se tes algún comentario, suxestión ou detectas algún problema durante a demostración, ponte en contacto connosco."

    voice = TTSVoice.load(model_path=model, config_path=config,
                          use_cuda=False, lang_code="gl-ES")
    print("-", voice.config.phoneme_type)
    print(voice.config)
    phones = voice.phonemize(sentence)
    print(phones)
    print(voice.phonemes_to_ids(phones[0]))

    slug = f"vits_{voice.config.phoneme_type.value}_{voice.config.lang_code}"
    with wave.open(f"{slug}.wav", "wb") as wav_file:
        voice.synthesize_wav(sentence, wav_file, syn_config)