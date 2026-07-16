"""Multi-voice discovery and switching."""
import json
import os

from conftest import wait_for_done


def _add_voice(directory, voice_id, config=None):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, f"{voice_id}.onnx"), "wb") as f:
        f.write(b"onnx")
    with open(os.path.join(directory, f"{voice_id}.onnx.json"), "w") as f:
        json.dump(config or {}, f)


class TestDiscovery:
    def test_finds_bundled_and_cache_voices(self, nvda_env):
        m = nvda_env.module
        _add_voice(m.VOICE_CACHE_DIR, "extra_pt-PT")
        voices = m.discover_voices()
        assert set(voices) == {m.DEFAULT_VOICE_ID, "extra_pt-PT"}

    def test_default_voice_listed_first(self, nvda_env):
        m = nvda_env.module
        _add_voice(m.VOICE_CACHE_DIR, "aaa_en-US")
        assert next(iter(m.discover_voices())) == m.DEFAULT_VOICE_ID

    def test_onnx_without_config_skipped(self, nvda_env):
        m = nvda_env.module
        os.makedirs(m.VOICE_CACHE_DIR, exist_ok=True)
        with open(os.path.join(m.VOICE_CACHE_DIR, "orphan.onnx"), "wb") as f:
            f.write(b"onnx")
        assert "orphan" not in m.discover_voices()

    def test_language_from_config_overrides_filename(self, nvda_env):
        m = nvda_env.module
        _add_voice(m.VOICE_CACHE_DIR, "weirdname", config={"lang_code": "pt"})
        assert m.discover_voices()["weirdname"]["language"] == "pt"

    def test_language_from_filename_fallback(self, nvda_env):
        m = nvda_env.module
        _add_voice(m.VOICE_CACHE_DIR, "voice_fr-FR")
        assert m.discover_voices()["voice_fr-FR"]["language"] == "fr-FR"


class TestSwitching:
    def test_cache_voice_selectable_and_speaks(self, nvda_env):
        m = nvda_env.module
        _add_voice(m.VOICE_CACHE_DIR, "extra_pt-PT")
        driver = m.SynthDriver()
        try:
            assert driver._voice_loaded_event.wait(5)
            assert "extra_pt-PT" in driver.availableVoices
            driver._set_voice("extra_pt-PT")
            assert driver._get_voice() == "extra_pt-PT"
            assert driver._voice_loaded_event.wait(5)
            driver.speak(["hello."])
            assert wait_for_done(nvda_env)
            assert len(driver._player.fed) > 0
        finally:
            driver.terminate()

    def test_switch_to_same_voice_is_noop(self, nvda_env):
        m = nvda_env.module
        driver = m.SynthDriver()
        try:
            assert driver._voice_loaded_event.wait(5)
            gen = driver._load_generation
            driver._set_voice(m.DEFAULT_VOICE_ID)
            assert driver._load_generation == gen
        finally:
            driver.terminate()
