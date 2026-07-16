"""Voice catalog and download management for the phoonnx NVDA add-on.

Pure logic, no NVDA or wx imports — the GUI in
``globalPlugins/phoonnxVoiceManager`` and the tests both drive this module.

Voices are installed as ``<id>.onnx`` + ``<id>.onnx.json`` pairs in the
phoonnx voice cache directory, where the synth driver's ``discover_voices()``
picks them up without reinstalling the add-on.
"""
import json
import os
import sys
import urllib.request
from typing import Callable, Dict, List, Optional

DRIVER_DIR = os.path.dirname(os.path.abspath(__file__))
PHOONNX_LIBS_PATH = os.path.join(DRIVER_DIR, "phoonnx_libs")
VOICE_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "phoonnx", "voices")

DOWNLOAD_TIMEOUT = 60


class VoiceManagerError(Exception):
    pass


def list_catalog(fetcher: Optional[Callable[[], List[dict]]] = None) -> List[Dict[str, str]]:
    """Return downloadable voices as ``{id, lang, model_url, config_url}`` dicts.

    ``fetcher`` overrides the default source (the bundled phoonnx voice
    index); entries missing a model or config URL are dropped, as the add-on
    can only load self-contained ``.onnx`` + ``.onnx.json`` voices.
    """
    entries = (fetcher or _phoonnx_catalog)()
    catalog = []
    for entry in entries:
        if not entry.get("model_url") or not entry.get("config_url"):
            continue
        catalog.append({
            "id": entry["id"],
            "lang": entry.get("lang") or "",
            "model_url": entry["model_url"],
            "config_url": entry["config_url"],
        })
    catalog.sort(key=lambda e: (e["lang"], e["id"]))
    return catalog


def _phoonnx_catalog() -> List[dict]:
    """Read the voice index bundled with the vendored phoonnx runtime."""
    if PHOONNX_LIBS_PATH not in sys.path and os.path.isdir(PHOONNX_LIBS_PATH):
        sys.path.insert(0, PHOONNX_LIBS_PATH)
    try:
        from phoonnx.model_manager import TTSModelManager
    except Exception as e:
        raise VoiceManagerError(f"phoonnx runtime not available: {e}")
    manager = TTSModelManager()
    try:
        manager.load()
        if not manager.all_voices:
            manager.get_ovos_voice_list()
            manager.get_piper_voice_list()
            manager.save()
    except Exception as e:
        raise VoiceManagerError(f"could not load the phoonnx voice catalog: {e}")
    return [
        {
            "id": v.voice_id,
            "lang": v.lang,
            "model_url": v.model_url,
            "config_url": v.config_url,
        }
        for v in manager.all_voices
    ]


def installed_voices(cache_dir: str = None) -> List[str]:
    """Voice ids installed in the cache dir (``<id>.onnx`` + ``<id>.onnx.json``)."""
    cache_dir = cache_dir or VOICE_CACHE_DIR
    if not os.path.isdir(cache_dir):
        return []
    voices = []
    for fn in sorted(os.listdir(cache_dir)):
        if fn.endswith(".onnx") and os.path.exists(os.path.join(cache_dir, fn + ".json")):
            voices.append(fn[:-len(".onnx")])
    return voices


def _download(url: str, dest: str,
              progress: Optional[Callable[[int, int], None]] = None,
              opener: Optional[Callable] = None):
    """Download ``url`` to ``dest`` atomically (``.part`` + rename)."""
    opener = opener or (lambda u: urllib.request.urlopen(u, timeout=DOWNLOAD_TIMEOUT))
    part = dest + ".part"
    try:
        with opener(url) as resp, open(part, "wb") as f:
            total = int(resp.headers.get("Content-Length") or 0) if hasattr(resp, "headers") else 0
            done = 0
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
        os.replace(part, dest)
    except Exception:
        if os.path.exists(part):
            os.remove(part)
        raise


def download_voice(entry: Dict[str, str], cache_dir: str = None,
                   progress: Optional[Callable[[int, int], None]] = None,
                   opener: Optional[Callable] = None) -> str:
    """Install a catalog entry into the voice cache; returns the model path.

    The config is fetched first so a failed model download never leaves a
    half-installed voice that ``discover_voices()`` would list.
    """
    cache_dir = cache_dir or VOICE_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    voice_id = entry["id"].replace("/", "_").replace("\\", "_")
    model_path = os.path.join(cache_dir, f"{voice_id}.onnx")
    config_path = model_path + ".json"
    try:
        _download(entry["config_url"], config_path, opener=opener)
        with open(config_path, encoding="utf-8") as f:
            json.load(f)
        _download(entry["model_url"], model_path, progress=progress, opener=opener)
    except json.JSONDecodeError:
        os.remove(config_path)
        raise VoiceManagerError(f"{entry['id']}: config URL did not return valid JSON")
    except Exception as e:
        for p in (config_path, model_path):
            if os.path.exists(p):
                os.remove(p)
        if isinstance(e, VoiceManagerError):
            raise
        raise VoiceManagerError(f"{entry['id']}: download failed: {e}")
    return model_path


def remove_voice(voice_id: str, cache_dir: str = None):
    """Delete an installed voice from the cache dir (bundled voices are not removable)."""
    cache_dir = cache_dir or VOICE_CACHE_DIR
    model_path = os.path.join(cache_dir, f"{voice_id}.onnx")
    if not os.path.exists(model_path):
        raise VoiceManagerError(f"{voice_id} is not installed in {cache_dir}")
    os.remove(model_path)
    config_path = model_path + ".json"
    if os.path.exists(config_path):
        os.remove(config_path)
