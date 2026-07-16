"""Voice catalog listing, download and removal (voice_manager.py)."""
import importlib.util
import io
import json
import os

import pytest

VM_PATH = os.path.join(os.path.dirname(__file__), "..",
                       "synthDrivers", "phoonnx", "voice_manager.py")


@pytest.fixture
def vm():
    spec = importlib.util.spec_from_file_location("voice_manager_under_test", VM_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_opener(payloads, fail_urls=()):
    """Fake urlopen: url -> bytes; raises for urls in fail_urls."""

    class FakeResponse(io.BytesIO):
        headers = {"Content-Length": "0"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.close()

    def opener(url):
        if url in fail_urls:
            raise OSError(f"connection refused: {url}")
        return FakeResponse(payloads[url])

    return opener


ENTRY = {
    "id": "nice_pt-PT",
    "lang": "pt-PT",
    "model_url": "http://x/model.onnx",
    "config_url": "http://x/model.onnx.json",
}
PAYLOADS = {
    "http://x/model.onnx": b"onnx-bytes",
    "http://x/model.onnx.json": json.dumps({"lang_code": "pt"}).encode(),
}


class TestCatalog:
    def test_list_catalog_filters_and_sorts(self, vm):
        entries = [
            {"id": "b", "lang": "en", "model_url": "m", "config_url": "c"},
            {"id": "no-config", "lang": "en", "model_url": "m", "config_url": None},
            {"id": "no-model", "lang": "en", "model_url": "", "config_url": "c"},
            {"id": "a", "lang": "de", "model_url": "m", "config_url": "c"},
        ]
        catalog = vm.list_catalog(fetcher=lambda: entries)
        assert [e["id"] for e in catalog] == ["a", "b"]

    def test_fetcher_error_propagates(self, vm):
        def boom():
            raise vm.VoiceManagerError("offline")
        with pytest.raises(vm.VoiceManagerError):
            vm.list_catalog(fetcher=boom)


class TestDownload:
    def test_download_installs_pair(self, vm, tmp_path):
        cache = str(tmp_path / "voices")
        model = vm.download_voice(ENTRY, cache_dir=cache, opener=make_opener(PAYLOADS))
        assert os.path.exists(model)
        assert os.path.exists(model + ".json")
        assert vm.installed_voices(cache) == ["nice_pt-PT"]

    def test_progress_reported(self, vm, tmp_path):
        calls = []
        vm.download_voice(ENTRY, cache_dir=str(tmp_path),
                          progress=lambda d, t: calls.append((d, t)),
                          opener=make_opener(PAYLOADS))
        assert calls and calls[-1][0] == len(PAYLOADS["http://x/model.onnx"])

    def test_failed_model_download_leaves_nothing(self, vm, tmp_path):
        cache = str(tmp_path)
        opener = make_opener(PAYLOADS, fail_urls={"http://x/model.onnx"})
        with pytest.raises(vm.VoiceManagerError):
            vm.download_voice(ENTRY, cache_dir=cache, opener=opener)
        assert vm.installed_voices(cache) == []
        assert not os.listdir(cache)

    def test_invalid_config_json_rejected(self, vm, tmp_path):
        payloads = dict(PAYLOADS)
        payloads["http://x/model.onnx.json"] = b"not json"
        with pytest.raises(vm.VoiceManagerError, match="valid JSON"):
            vm.download_voice(ENTRY, cache_dir=str(tmp_path), opener=make_opener(payloads))
        assert not os.listdir(str(tmp_path))

    def test_voice_id_path_characters_sanitized(self, vm, tmp_path):
        entry = dict(ENTRY, id="org/evil_en-US")
        model = vm.download_voice(entry, cache_dir=str(tmp_path), opener=make_opener(PAYLOADS))
        assert os.path.basename(model) == "org_evil_en-US.onnx"
        assert os.path.dirname(model) == str(tmp_path)


class TestRemove:
    def test_remove_deletes_pair(self, vm, tmp_path):
        cache = str(tmp_path)
        vm.download_voice(ENTRY, cache_dir=cache, opener=make_opener(PAYLOADS))
        vm.remove_voice("nice_pt-PT", cache_dir=cache)
        assert vm.installed_voices(cache) == []
        assert not os.listdir(cache)

    def test_remove_missing_raises(self, vm, tmp_path):
        with pytest.raises(vm.VoiceManagerError):
            vm.remove_voice("ghost", cache_dir=str(tmp_path))


class TestInstalled:
    def test_missing_dir_is_empty(self, vm, tmp_path):
        assert vm.installed_voices(str(tmp_path / "nope")) == []

    def test_orphan_onnx_ignored(self, vm, tmp_path):
        (tmp_path / "orphan.onnx").write_bytes(b"x")
        assert vm.installed_voices(str(tmp_path)) == []
