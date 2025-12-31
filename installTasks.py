# installTasks.py
import os
import sys
import json
import traceback
import languageHandler  # For reading the NVDA interface language
from logHandler import log
from pathlib import Path

def onInstall():
    addon_dir = os.path.dirname(os.path.abspath(__file__))
    libs_path = os.path.join(addon_dir, "phoonnx_libs")
    user_home = os.path.expanduser("~")

    # 1. Determine the NVDA language code
    try:
        raw_lang = languageHandler.getLanguage()
        # Normalize: NVDA often returns 'nl' or 'en_US'. 
        # We convert it to 'nl-NL' or 'en-US' for better compatibility with voice indices.
        if "_" in raw_lang:
            target_lang = raw_lang.replace("_", "-")
        elif raw_lang == "nl":
            target_lang = "nl-NL"
        elif raw_lang == "en":
            target_lang = "en-US"
        else:
            target_lang = raw_lang
    except Exception:
        target_lang = "en-US" # Safe fallback

    if libs_path not in sys.path:
        sys.path.insert(0, libs_path)
    
    target_cache_dir = os.path.join(user_home, ".cache", "phoonnx")
    voice_index_dir = os.path.join(libs_path, 'phoonnx', 'voice_index')
    voice_manager_state_file = os.path.join(target_cache_dir, "voices_cache.json")

    try:
        if not os.path.exists(target_cache_dir):
            os.makedirs(target_cache_dir, exist_ok=True)

        from phoonnx.model_manager import TTSModelManager, TTSModelInfo
        manager = TTSModelManager(cache_path=voice_manager_state_file)
        manager.models_dir = target_cache_dir

        # --- OPTIMIZED: No longer scanning all JSON files during installation ---
        # Full indexing now only happens when the settings panel is opened.

        ovos_index_path = os.path.join(voice_index_dir, "OVOS.json")
        if not os.path.exists(ovos_index_path):
            return

        with open(ovos_index_path, 'r', encoding='utf-8') as f:
            voices_data = json.load(f)

        # 2. Find a voice that matches the target_lang
        target_voice_id = None
        for voice_id, info in voices_data.items():
            if info.get("lang") == target_lang:
                target_voice_id = voice_id
                break
        
        # Fallback if the specific region (e.g., nl-BE) does not exist: try the main language (nl)
        if not target_voice_id:
            short_lang = target_lang.split('-')[0]
            for voice_id, info in voices_data.items():
                if info.get("lang", "").startswith(short_lang):
                    target_voice_id = voice_id
                    break

        # Ultimate fallback to a known voice if nothing else is found
        if not target_voice_id:
            target_voice_id = "OpenVoiceOS/pipertts_en-GB_southern_english_female-low"

        data = voices_data.get(target_voice_id)

        if data:
            m_url = data.get("model_url")
            match_info = TTSModelInfo(
                voice_id=target_voice_id,
                lang=data.get("lang"),
                model_url=m_url,
                config_url=data.get("config_url") or m_url.replace(".onnx", ".json")
            )
            
            match_info.download_model()
            match_info.download_config()
            
            onnx_check = os.path.join(str(match_info.voice_path), "model.onnx")
            if os.path.exists(onnx_check):
                log.info(f"Phoonnx Installer: {target_voice_id} successfully downloaded.")
    except Exception as e:
        log.error(f"Phoonnx Installer Error: {str(e)}")