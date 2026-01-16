# installTasks.py
import os
import sys
import json
import subprocess
import urllib.request
import shutil
import languageHandler
import gui
import wx
from logHandler import log
from pathlib import Path

def find_existing_espeak():
    """Checks if espeak-ng.exe is already present on the system."""
    exe_name = "espeak-ng.exe"
    path_env = os.environ.get("PATH", "").split(os.pathsep)
    for folder in path_env:
        potential_path = os.path.join(folder, exe_name)
        if os.path.exists(potential_path):
            return potential_path

    program_files = [
        os.environ.get("ProgramFiles", "C:\\Program Files"),
        os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")
    ]
    for pf in program_files:
        potential_path = os.path.join(pf, "eSpeak NG", exe_name)
        if os.path.exists(potential_path):
            return potential_path
    return None

def onInstall():
    addon_dir = os.path.dirname(os.path.abspath(__file__))
    libs_path = os.path.join(addon_dir, "phoonnx_libs")
    bin_dir = os.path.join(addon_dir, "bin")
    
    # --- STEP 0: PREPARE CACHE & SYNC BUNDLED DATA ---
    user_home = os.path.expanduser("~")
    phoonnx_cache_dir = os.path.join(user_home, ".cache", "phoonnx")
    
    if not os.path.exists(phoonnx_cache_dir):
        os.makedirs(phoonnx_cache_dir, exist_ok=True)

    try:
        bundled_voices_dir = os.path.join(addon_dir, "voices")
        bundled_json = os.path.join(addon_dir, "voices.json")
        dest_voices_dir = os.path.join(phoonnx_cache_dir, "voices")
        dest_json = os.path.join(phoonnx_cache_dir, "voices.json")

        # FIX: The SettingsPanel needs the 'voices' directory to exist in cache
        if os.path.exists(bundled_voices_dir):
            if not os.path.exists(dest_voices_dir):
                shutil.copytree(bundled_voices_dir, dest_voices_dir)
            else:
                # Minimal sync of missing bundled configs
                for item in os.listdir(bundled_voices_dir):
                    s = os.path.join(bundled_voices_dir, item)
                    d = os.path.join(dest_voices_dir, item)
                    if not os.path.exists(d):
                        if os.path.isdir(s): shutil.copytree(s, d)
                        else: shutil.copy2(s, d)
        
        if os.path.exists(bundled_json):
            shutil.copy2(bundled_json, dest_json)
    except Exception as e:
        log.error(f"Phoonnx Installer: Error syncing bundled data: {e}")

    # --- PART 1: ESPEAK-NG INSTALLATION ---
    if not os.path.exists(bin_dir):
        os.makedirs(bin_dir)
    target_exe_path = os.path.join(bin_dir, "espeak-ng.exe")
    target_dll_path = os.path.join(bin_dir, "libespeak-ng.dll")
    target_data_dir = os.path.join(bin_dir, "espeak-ng-data")

    if not os.path.exists(target_exe_path) and not find_existing_espeak():
        msg = "Phoonnx requires eSpeak-NG components. Would you like to download them now?"
        if gui.messageBox(msg, "Installation Required", wx.YES_NO | wx.ICON_QUESTION) == wx.YES:
            try:
                msi_url = "https://github.com/espeak-ng/espeak-ng/releases/download/1.51/espeak-ng-X64.msi"
                msi_path = os.path.join(addon_dir, "espeak-setup.msi")
                temp_extract_dir = os.path.join(addon_dir, "temp_msi_extract")

                with urllib.request.urlopen(msi_url) as response, open(msi_path, 'wb') as out_file:
                    out_file.write(response.read())

                subprocess.run(f'msiexec /a "{msi_path}" /qb TARGETDIR="{temp_extract_dir}"', check=True, shell=True)

                for root, dirs, files in os.walk(temp_extract_dir):
                    if "espeak-ng.exe" in files:
                        shutil.copy2(os.path.join(root, "espeak-ng.exe"), target_exe_path)
                    if "libespeak-ng.dll" in files:
                        shutil.copy2(os.path.join(root, "libespeak-ng.dll"), target_dll_path)
                    if root.endswith("espeak-ng-data"):
                        if os.path.exists(target_data_dir): shutil.rmtree(target_data_dir)
                        shutil.copytree(root, target_data_dir)
                
                if os.path.exists(msi_path): os.remove(msi_path)
                shutil.rmtree(temp_extract_dir, ignore_errors=True)
            except Exception as e:
                log.error(f"Phoonnx Installer: eSpeak-NG install failed: {e}")

    # --- PART 2: FAST VOICE SELECTION ---
    if libs_path not in sys.path:
        sys.path.insert(0, libs_path)

    try:
        from phoonnx.model_manager import TTSModelInfo
        
        try:
            raw_lang = languageHandler.getLanguage()
            target_lang = raw_lang.replace("_", "-") if "_" in raw_lang else raw_lang
            if target_lang == "nl": target_lang = "nl-NL"
            elif target_lang == "en": target_lang = "en-US"
        except:
            target_lang = "en-US"

        selected_voice_data = None
        if os.path.exists(dest_json):
            with open(dest_json, 'r', encoding='utf-8') as f:
                voices_dict = json.load(f)
                
                for vid, vdata in voices_dict.items():
                    if vdata.get('lang') == target_lang:
                        selected_voice_data = vdata
                        break
                
                if not selected_voice_data:
                    short_lang = target_lang.split('-')[0]
                    for vid, vdata in voices_dict.items():
                        if vdata.get('lang', '').startswith(short_lang):
                            selected_voice_data = vdata
                            break

        if selected_voice_data:
            selected_voice = TTSModelInfo(**selected_voice_data)
            v_id_fixed = selected_voice.voice_id.replace("/", os.sep)
            model_path = os.path.join(phoonnx_cache_dir, "voices", v_id_fixed, "model.onnx")
            
            if not os.path.exists(model_path):
                log.info(f"Phoonnx Installer: Downloading model for {selected_voice.voice_id}...")
                try:
                    selected_voice.download_config() 
                    selected_voice.download_model()
                    if getattr(selected_voice, 'tokens_url', None) or getattr(selected_voice, 'phoneme_map_url', None):
                        selected_voice.download_phoneme_map()
                    log.info("Phoonnx Installer: Initial download complete.")
                except Exception as e:
                    log.warning(f"Phoonnx Installer: Download failed: {e}")
            else:
                log.info(f"Phoonnx Installer: Model {selected_voice.voice_id} already present.")
                
    except Exception as e:
        log.error(f"Phoonnx Installer: Voice setup failed: {e}")

if __name__ == "__main__":
    onInstall()
