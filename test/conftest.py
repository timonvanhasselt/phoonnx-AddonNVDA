"""Test fixtures for the phoonnx NVDA synth driver.

NVDA host modules (nvwave, logHandler, synthDriverHandler, speech.commands)
and the phoonnx runtime are stubbed so the driver can be imported and
exercised on any platform, without NVDA or ONNX models.
"""
import importlib.util
import os
import sys
import threading
import types

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVER_PATH = os.path.join(REPO_ROOT, "synthDrivers", "phoonnx", "__init__.py")


# ---------------------------------------------------------------------------
# NVDA host stubs
# ---------------------------------------------------------------------------

class FakeWavePlayer:
    def __init__(self, channels, samplesPerSec, bitsPerSample, purpose=None):
        self.channels = channels
        self.samplesPerSec = samplesPerSec
        self.bitsPerSample = bitsPerSample
        self.purpose = purpose
        self.fed = []
        self.stopped = False
        self.paused = None
        self.closed = False
        self.idled = False

    def feed(self, chunk):
        self.fed.append(chunk)

    def stop(self):
        self.stopped = True

    def pause(self, switch):
        self.paused = switch

    def close(self):
        self.closed = True

    def idle(self):
        self.idled = True


def _install_nvda_stubs():
    nvwave = types.ModuleType("nvwave")
    nvwave.WavePlayer = FakeWavePlayer
    nvwave.AudioPurpose = types.SimpleNamespace(SPEECH="speech")
    sys.modules["nvwave"] = nvwave

    logHandler = types.ModuleType("logHandler")

    class _Log:
        def _noop(self, *a, **k):
            pass
        debug = info = warning = error = critical = _noop

    logHandler.log = _Log()
    sys.modules["logHandler"] = logHandler

    synthDriverHandler = types.ModuleType("synthDriverHandler")

    class VoiceInfo:
        def __init__(self, id, displayName, language=None):
            self.id = id
            self.displayName = displayName
            self.language = language

    class _Notifier:
        def __init__(self):
            self.notifications = []

        def notify(self, **kwargs):
            self.notifications.append(kwargs)

    class _Setting:
        def __init__(self, *a, **k):
            pass

    class BaseSynthDriver:
        VoiceSetting = _Setting
        RateSetting = _Setting
        VolumeSetting = _Setting
        PitchSetting = _Setting

        def __init__(self):
            pass

        @property
        def availableVoices(self):
            return self._getAvailableVoices()

    synthDriverHandler.SynthDriver = BaseSynthDriver
    synthDriverHandler.VoiceInfo = VoiceInfo
    synthDriverHandler.synthIndexReached = _Notifier()
    synthDriverHandler.synthDoneSpeaking = _Notifier()
    sys.modules["synthDriverHandler"] = synthDriverHandler

    speech = types.ModuleType("speech")
    commands = types.ModuleType("speech.commands")

    class IndexCommand:
        def __init__(self, index):
            self.index = index

    class RateCommand:
        def __init__(self, value=50):
            self.value = value

    class PitchCommand:
        def __init__(self, value=50):
            self.value = value

    class VolumeCommand:
        def __init__(self, value=100):
            self.value = value

    class BreakCommand:
        def __init__(self, time=0):
            self.time = time

    for cls in (IndexCommand, RateCommand, PitchCommand, VolumeCommand, BreakCommand):
        setattr(commands, cls.__name__, cls)
    speech.commands = commands
    sys.modules["speech"] = speech
    sys.modules["speech.commands"] = commands

    addonHandler = types.ModuleType("addonHandler")
    addonHandler.initTranslation = lambda: None
    sys.modules["addonHandler"] = addonHandler
    import builtins
    builtins._ = lambda s: s
    return synthDriverHandler


# ---------------------------------------------------------------------------
# phoonnx runtime stubs
# ---------------------------------------------------------------------------

class FakeSynthesisConfig:
    def __init__(self, length_scale=1.0, noise_scale=None, noise_w_scale=None,
                 enable_phonetic_spellings=True, add_diacritics=False, volume=1.0):
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w_scale = noise_w_scale
        self.enable_phonetic_spellings = enable_phonetic_spellings
        self.add_diacritics = add_diacritics
        self.volume = volume


class FakeTTSVoice:
    """Produces a deterministic ramp of audio per phoneme-id list."""

    samples_per_sentence = 10000

    def __init__(self):
        self.config = types.SimpleNamespace(sample_rate=22050, lang_code="nl")
        self.phonetic_spellings = None
        self.phonemizer = None
        self.synth_calls = []

    @classmethod
    def load(cls, model_path, config_path):
        return cls()

    def phonemize(self, text):
        # one "sentence" per period
        return [list(s.strip()) for s in text.split(".") if s.strip()]

    def phonemes_to_ids(self, phonemes):
        return [ord(p) for p in phonemes]

    def phoneme_ids_to_audio(self, phoneme_ids, config):
        self.synth_calls.append((list(phoneme_ids), config))
        return np.linspace(-0.5, 0.5, self.samples_per_sentence, dtype=np.float32)


def _install_phoonnx_stubs():
    phoonnx = types.ModuleType("phoonnx")
    config_mod = types.ModuleType("phoonnx.config")
    config_mod.SynthesisConfig = FakeSynthesisConfig
    voice_mod = types.ModuleType("phoonnx.voice")
    voice_mod.TTSVoice = FakeTTSVoice

    class _Log:
        def _noop(self, *a, **k):
            pass
        debug = info = warning = error = critical = _noop

    voice_mod.LOG = _Log()
    phoonnx.config = config_mod
    phoonnx.voice = voice_mod
    sys.modules["phoonnx"] = phoonnx
    sys.modules["phoonnx.config"] = config_mod
    sys.modules["phoonnx.voice"] = voice_mod


@pytest.fixture()
def nvda_env(tmp_path):
    """Install stubs, import the driver fresh, point it at dummy model files."""
    saved_modules = dict(sys.modules)
    synthDriverHandler = _install_nvda_stubs()
    _install_phoonnx_stubs()

    spec = importlib.util.spec_from_file_location("nvda_phoonnx_driver", DRIVER_PATH)
    driver_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver_mod)

    # dummy voice files so check() passes
    (tmp_path / driver_mod.MODEL_FILENAME).write_bytes(b"onnx")
    (tmp_path / driver_mod.CONFIG_FILENAME).write_text("{}")
    driver_mod.DRIVER_DIR = str(tmp_path)

    yield types.SimpleNamespace(
        module=driver_mod,
        synthDriverHandler=synthDriverHandler,
        commands=sys.modules["speech.commands"],
    )

    # tear down any driver threads and restore modules
    sys.modules.clear()
    sys.modules.update(saved_modules)


@pytest.fixture()
def driver(nvda_env):
    d = nvda_env.module.SynthDriver()
    assert d._worker_thread is not None
    # wait for async voice load
    assert nvda_env.module and d._voice_loaded_event.wait(5)
    yield d
    d.terminate()


def wait_for_done(nvda_env, count=1, timeout=5.0):
    """Wait until synthDoneSpeaking has been notified `count` times."""
    notifier = nvda_env.synthDriverHandler.synthDoneSpeaking
    deadline = threading.Event()
    for _ in range(int(timeout / 0.02)):
        if len(notifier.notifications) >= count:
            return True
        deadline.wait(0.02)
    return False
