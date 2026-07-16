"""Unit tests for pure helpers and the PatchedVoice wrapper."""
import numpy as np


class TestRateMapping:
    def test_default_rate_is_normal_speed(self, nvda_env):
        assert nvda_env.module.nvda_rate_to_length_scale(50) == 1.0

    def test_rate_zero_does_not_divide_by_zero(self, nvda_env):
        # regression: rate 0 used to raise ZeroDivisionError
        assert nvda_env.module.nvda_rate_to_length_scale(0) == 2.0

    def test_extremes_are_clamped(self, nvda_env):
        f = nvda_env.module.nvda_rate_to_length_scale
        assert f(100) == 0.5
        assert f(1) == 2.0
        assert f(-5) == 2.0
        assert f(1000) == 0.5

    def test_faster_rate_gives_smaller_scale(self, nvda_env):
        f = nvda_env.module.nvda_rate_to_length_scale
        assert f(80) < f(50) < f(20)


class TestSplitSpeechSequence:
    def test_plain_text(self, nvda_env):
        segs, rate = nvda_env.module.split_speech_sequence(["hello world"], 50)
        assert segs == [("text", "hello world")]
        assert rate == 50

    def test_index_commands_keep_position(self, nvda_env):
        c = nvda_env.commands
        segs, _ = nvda_env.module.split_speech_sequence(
            ["one", c.IndexCommand(7), "two"], 50)
        assert segs == [("text", "one"), ("index", 7), ("text", "two")]

    def test_break_command(self, nvda_env):
        c = nvda_env.commands
        segs, _ = nvda_env.module.split_speech_sequence(
            ["a", c.BreakCommand(time=250)], 50)
        assert segs == [("text", "a"), ("break", 250)]

    def test_rate_command_overrides_rate(self, nvda_env):
        c = nvda_env.commands
        _, rate = nvda_env.module.split_speech_sequence([c.RateCommand(80), "x"], 50)
        assert rate == 80

    def test_whitespace_only_text_dropped(self, nvda_env):
        segs, _ = nvda_env.module.split_speech_sequence(["   "], 50)
        assert segs == []

    def test_pitch_and_volume_commands_ignored(self, nvda_env):
        c = nvda_env.commands
        segs, _ = nvda_env.module.split_speech_sequence(
            [c.PitchCommand(10), "hi", c.VolumeCommand(30)], 50)
        assert segs == [("text", "hi")]


class TestPatchedVoice:
    def _voice(self, nvda_env):
        PatchedVoice, SynthesisConfig = nvda_env.module.import_phoonnx()
        assert PatchedVoice is not None
        return PatchedVoice.load("model", "config"), SynthesisConfig

    def test_streams_int16_chunks(self, nvda_env):
        voice, SynthesisConfig = self._voice(nvda_env)
        chunks = []
        ok = voice.synthesize_to_callback(
            "hello.", lambda c: chunks.append(c) and 0, config=SynthesisConfig())
        assert ok
        total = sum(len(c) for c in chunks)
        assert total == voice._original_voice.samples_per_sentence * 2  # int16
        assert all(len(c) <= nvda_env.module.CHUNK_SIZE for c in chunks)

    def test_audio_is_normalized(self, nvda_env):
        voice, SynthesisConfig = self._voice(nvda_env)
        chunks = []
        voice.synthesize_to_callback("hi.", lambda c: chunks.append(c) and 0,
                                     config=SynthesisConfig())
        audio = np.frombuffer(b"".join(chunks), dtype=np.int16)
        # source ramp peaks at 0.5 but is normalized to full range
        assert abs(int(audio.max())) > 30000

    def test_volume_scales_audio(self, nvda_env):
        voice, SynthesisConfig = self._voice(nvda_env)
        loud, quiet = [], []
        voice.synthesize_to_callback("hi.", lambda c: loud.append(c) and 0,
                                     config=SynthesisConfig(volume=1.0))
        voice.synthesize_to_callback("hi.", lambda c: quiet.append(c) and 0,
                                     config=SynthesisConfig(volume=0.5))
        a_loud = np.frombuffer(b"".join(loud), dtype=np.int16)
        a_quiet = np.frombuffer(b"".join(quiet), dtype=np.int16)
        assert a_quiet.max() < a_loud.max() * 0.6

    def test_abort_via_callback_return(self, nvda_env):
        voice, SynthesisConfig = self._voice(nvda_env)
        seen = []

        def cb(chunk):
            seen.append(chunk)
            return 1  # abort immediately

        ok = voice.synthesize_to_callback("hello. world.", cb, config=SynthesisConfig())
        assert ok is False
        assert len(seen) == 1

    def test_synthesis_error_returns_false(self, nvda_env):
        voice, SynthesisConfig = self._voice(nvda_env)
        voice._original_voice.phoneme_ids_to_audio = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        ok = voice.synthesize_to_callback("hi.", lambda c: 0, config=SynthesisConfig())
        assert ok is False

    def test_multiple_sentences(self, nvda_env):
        voice, SynthesisConfig = self._voice(nvda_env)
        voice.synthesize_to_callback("one. two. three.", lambda c: 0,
                                     config=SynthesisConfig())
        assert len(voice._original_voice.synth_calls) == 3
