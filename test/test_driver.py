"""End-to-end tests of SynthDriver against the stubbed NVDA/phoonnx runtime."""
import time

from conftest import wait_for_done


class TestCheck:
    def test_check_true_with_model_files(self, nvda_env):
        assert nvda_env.module.SynthDriver.check() is True

    def test_check_false_without_model(self, nvda_env, tmp_path):
        nvda_env.module.DRIVER_DIR = str(tmp_path / "empty")
        assert nvda_env.module.SynthDriver.check() is False


class TestVoices:
    def test_available_voices(self, driver, nvda_env):
        voices = driver.availableVoices
        assert nvda_env.module.DEFAULT_VOICE_ID in voices
        assert voices[nvda_env.module.DEFAULT_VOICE_ID].language == "nl-NL"

    def test_set_invalid_voice_ignored(self, driver):
        before = driver._get_voice()
        driver._set_voice("nonexistent")
        assert driver._get_voice() == before

    def test_language(self, driver):
        assert driver._get_language() == "nl-NL"
        assert "nl-NL" in driver._get_availableLanguages()


class TestSettings:
    def test_rate_volume_roundtrip(self, driver):
        driver._set_rate(80)
        assert driver._get_rate() == 80
        driver._set_volume(30)
        assert driver._get_volume() == 30


class TestSpeak:
    def test_speak_feeds_player_and_notifies_done(self, driver, nvda_env):
        driver.speak(["hello world."])
        assert wait_for_done(nvda_env)
        assert driver._player is not None
        assert len(driver._player.fed) > 0
        assert driver._player.idled  # audio flushed

    def test_speak_does_not_block(self, driver):
        start = time.monotonic()
        driver.speak(["hello."])
        assert time.monotonic() - start < 0.5

    def test_index_commands_notified(self, driver, nvda_env):
        c = nvda_env.commands
        driver.speak(["one.", c.IndexCommand(42), "two."])
        assert wait_for_done(nvda_env)
        indexes = [n["index"] for n in
                   nvda_env.synthDriverHandler.synthIndexReached.notifications]
        assert 42 in indexes

    def test_break_command_inserts_silence(self, driver, nvda_env):
        c = nvda_env.commands
        driver.speak([c.BreakCommand(time=100)])
        assert wait_for_done(nvda_env)
        fed = b"".join(driver._player.fed)
        expected = int(22050 * 0.1) * 2
        assert len(fed) == expected
        assert fed == bytes(expected)

    def test_volume_setting_applied(self, driver, nvda_env):
        driver._set_volume(40)
        driver.speak(["hi."])
        assert wait_for_done(nvda_env)
        # FakeTTSVoice records the config used
        voice = driver.tts_voice._original_voice
        _, config = voice.synth_calls[-1]
        assert config.volume == 0.4

    def test_rate_setting_applied(self, driver, nvda_env):
        driver._set_rate(100)
        driver.speak(["hi."])
        assert wait_for_done(nvda_env)
        voice = driver.tts_voice._original_voice
        _, config = voice.synth_calls[-1]
        assert config.length_scale == 0.5

    def test_rate_zero_speaks_without_crash(self, driver, nvda_env):
        driver._set_rate(0)
        driver.speak(["hi."])
        assert wait_for_done(nvda_env)
        assert len(driver._player.fed) > 0

    def test_empty_sequence_is_noop(self, driver, nvda_env):
        driver.speak([])
        driver.speak(["   "])
        time.sleep(0.2)
        assert not nvda_env.synthDriverHandler.synthDoneSpeaking.notifications


class TestControl:
    def test_cancel_stops_player_and_drains_queue(self, driver):
        for _ in range(5):
            driver.speak(["a very long sentence. " * 10])
        driver.cancel()
        assert driver._player.stopped
        assert driver._request_queue.empty()

    def test_pause_forwards_to_player(self, driver):
        driver.pause(True)
        assert driver._player.paused is True
        driver.pause(False)
        assert driver._player.paused is False

    def test_terminate_shuts_down_worker(self, driver):
        player = driver._player
        driver.terminate()
        assert player.closed
        assert driver.tts_voice is None
        driver._worker_thread.join(timeout=2)
        assert not driver._worker_thread.is_alive()
