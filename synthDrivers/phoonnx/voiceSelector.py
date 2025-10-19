# voiceSelector.py

# -*- coding: UTF-8 -*-

import os
import sys
import threading
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import shutil # Needed for the move operation

import config
import gui
import wx
from logHandler import log
# synthDriverHandler is still needed for other parts of the driver, but we
# use the config module directly for the synth configuration.
import synthDriverHandler 

# --- CRUCIAL CODE TO RESOLVE THE IMPORT ERROR ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ADDON_ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR)) 

if ADDON_ROOT_DIR not in sys.path:
    sys.path.insert(0, ADDON_ROOT_DIR)
# ---------------------------------------------------

# Import the necessary classes from model_manager
from phoonnx import model_manager 
from phoonnx.model_manager import TTSModelManager, TTSModelInfo


# Define the directory where voices are stored locally (within the NVDA cache)
VOICE_INSTALL_DIR = Path(ADDON_ROOT_DIR) / 'synthDrivers' / 'phoonnx' / 'voices'


def _get_manager() -> TTSModelManager:
    """Helper to initialize the model manager and load the local cache."""
    manager = TTSModelManager()
    manager.load()
    return manager


def get_nvda_language() -> str:
    """Returns the current NVDA language, e.g., 'nl' or 'en'."""
    try:
        lang = config.conf["general"]["language"]
        if "_" in lang:
            lang = lang.split("_")[0]
        return lang
    except Exception as e:
        log.error(f"Could not retrieve NVDA language: {e}")
        return "en"


def suggest_voice(voices: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Determines a suitable voice based on NVDA language."""
    nvda_lang = get_nvda_language()
    log.info(f"Preferred language: {nvda_lang}")

    for v in voices:
        if v.get("lang", "").startswith(nvda_lang):
            return v

    for v in voices:
        if v.get("lang", "").startswith("en"):
            return v

    return voices[0] if voices else None


def get_selected_voice() -> Tuple[Optional[str], Optional[str]]:
    """
    Reads the current Phoonnx voice ID and version from NVDA config.
    FIX: Uses config.conf directly.
    """
    try:
        # First, check if 'phoonnx' is the active synthesizer in the general config
        if config.conf["speech"]["synth"] != "phoonnx":
             return None, None
            
        # Now retrieve the 'voice' setting from the synth-specific configuration
        synth_config = config.conf["synthDrivers"]["phoonnx"]
        voice_id = synth_config.get("voice")
        
        if not voice_id:
            return None, None
            
        manager = _get_manager()
        local_config: Optional[TTSModelInfo] = manager.voices.get(voice_id)
        
        if local_config:
            # Use the voice_id as the "version" for comparison
            return voice_id, voice_id 
        
        return voice_id, None

    except Exception as e:
        # This catches the error if the 'voice' or 'phoonnx' key does not yet exist
        log.error(f"Error retrieving the selected voice: {e}")
        return None, None


def get_remote_voices_for_gui() -> List[Dict[str, Any]]:
    """Retrieves the remote voices via the TTSModelManager and formats them for the GUI."""
    manager = _get_manager()
    try:
        # This is the crucial network call
        manager.refresh_voices()
    except Exception as e:
        log.error(f"Error retrieving remote voices: {e}", exc_info=True)
        if manager.all_voices:
            log.warning("Refresh failed, using locally cached voice list.")
        else:
            raise 

    voices_for_gui = []
    
    for v_info in manager.all_voices: # manager.all_voices is List[TTSModelInfo]
        
        # Try to generate a readable name
        name = v_info.voice_id
        parts = v_info.voice_id.split('/')[-1].split('_')
        if len(parts) >= 2:
            name_parts = [p.capitalize() for p in parts if p.lower() not in ["pipertts", "phoonnx", "tugaphone", "espeak", "vits", "graphemes", "onnx", v_info.lang.lower()]]
            name = f"{' '.join(name_parts)} ({v_info.lang})"
        
        # Format the data for the NVDA/wxPython GUI dialog.
        voices_for_gui.append({
            "id": v_info.voice_id,
            "name": name.replace('_', ' ').strip(),
            "lang": v_info.lang,
            "version": v_info.voice_id, 
            "size_mb": 50, # Dummy or estimated
            
            # Add essential URLs so we can reconstruct the TTSModelInfo object later
            "config_url": v_info.config_url,
            "model_url": v_info.model_url,
            "tokens_url": v_info.tokens_url,
            "phoneme_map_url": v_info.phoneme_map_url,
            "phoneme_type": str(v_info.phoneme_type) if v_info.phoneme_type else None
        })

    return voices_for_gui


def download_voice(voice_info: Dict[str, Any]):
    """Downloads the selected voice, moves it, saves the configuration, and forces the driver to reload."""
    log.info(f"Starting download for voice: {voice_info.get('name', voice_info['id'])}")
    
    try:
        # Step 1: CONVERT the dictionary back to a TTSModelInfo object
        info = TTSModelInfo(
            voice_id=voice_info['id'],
            lang=voice_info['lang'],
            model_url=voice_info['model_url'],
            config_url=voice_info['config_url'],
            tokens_url=voice_info.get('tokens_url'),
            phoneme_map_url=voice_info.get('phoneme_map_url'),
        )
        
        log.info(f"Loading (downloading) TTSModelInfo. Temporary location: {info.voice_path}")
        # This is the function that initiates the network connection to download the model files
        info.load() 
        
        source_path = Path(info.voice_path)

        # Step 2: Determine the final destination in the addon folder and move
        target_path = VOICE_INSTALL_DIR / voice_info['id']
        
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if source_path.exists():
            shutil.move(source_path, target_path)
            log.info(f"Voice directory successfully moved to: {target_path}")
        else:
            raise FileNotFoundError(f"Download complete, but the expected source folder '{source_path}' does not exist. Cannot move.")

        # Update Manager Cache and save (to recognize local voices)
        manager = _get_manager()
        manager.add_voice(info)
        manager.save()
        manager.load() # Reload to recognize the moved voice
        log.info("TTSModelManager reloaded to register moved voice.")

        # Step 3: Save configuration change
        if config.conf["speech"]["synth"] == "phoonnx":
            
            # Ensure the subsection exists under [speech]
            if "phoonnx" not in config.conf["speech"]:
                 config.conf["speech"]["phoonnx"] = {} 
            
            # Set the 'voice' key
            config.conf["speech"]["phoonnx"]["voice"] = voice_info['id'] 
            
            # Save the configuration. This is CRUCIAL.
            config.conf.save() 
            log.info(f"Default voice '{voice_info['id']}' saved to config.")
            
            # --- CRUCIAL CORRECTION: Force reload by resetting the synthesizer ---
            
            def reload_synth_and_notify():
                # This function MUST run on the GUI thread.
                current_synth = "phoonnx"
                
                try:
                    # Resetting the synthesizer, so to speak.
                    # setSynth terminates the current driver and restarts it with the new settings.
                    log.info(f"Forcing reload by calling synthDriverHandler.setSynth('{current_synth}') again.")
                    synthDriverHandler.setSynth(current_synth, isFallback=False) 

                    # Notification
                    gui.messageBox(f"Voice '{voice_info.get('name', voice_info['id'])}' successfully downloaded. Choose insert + control + v to set the voice to default.", 
                                   "Success", 
                                   style=wx.ICON_INFORMATION)
                except Exception as e:
                    log.error(f"Error while resetting the synthesizer: {e}", exc_info=True)
                    # Show error message if changing the synth fails
                    gui.messageBox(f"Voice downloaded, but error setting it up: {e}", 
                                   "Synthesizer Error", 
                                   style=wx.ICON_ERROR)

            # Call the reload function on the GUI thread
            wx.CallAfter(reload_synth_and_notify)
        
    except Exception as e:
        log.error(f"Error downloading or installing voice: {e}", exc_info=True)
        wx.CallAfter(gui.messageBox,
                       f"An error occurred while downloading the voice:\n{e}", 
                       "Error", 
                       style=wx.ICON_ERROR)


def show_voice_selection_dialog(voices: List[Dict[str, Any]], suggestion: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Shows a dialog to select a voice using wx.SingleChoiceDialog."""
    
    choices = [f"{v.get('name', v['id'])} - {v.get('size_mb', '??')} MB" for v in voices]
    
    try:
        suggest_index = next(i for i, v in enumerate(voices) if v['id'] == suggestion['id'])
    except StopIteration:
        suggest_index = 0
    
    # Use wx.SingleChoiceDialog, which is available in the GUI thread
    dialog = wx.SingleChoiceDialog(
        parent=gui.mainFrame, 
        message="Select the desired voice to download and set as default:", 
        caption="Phoonnx Voice Selection", 
        choices=choices,
        style=wx.CAPTION | wx.SYSTEM_MENU | wx.CLOSE_BOX | wx.OK | wx.CANCEL
    )
    dialog.SetSelection(suggest_index)
    
    # --- BELANGRIJKE WIJZIGING: Focus geven aan het dialoogvenster ---
    # Raise() of CenterOnScreen() gevolgd door Raise() kan de focus garanderen.
    dialog.Raise()
    # -----------------------------------------------------------------
    
    selectedIndex = None
    if dialog.ShowModal() == wx.ID_OK:
        selectedIndex = dialog.GetSelection()
        
    dialog.Destroy()
    
    if selectedIndex is not None and selectedIndex >= 0:
        return voices[selectedIndex]
    return None


def run_voice_selection(force=False):
    """The main logic for the 'First Run' voice selection."""
    
    def run_in_thread():
        log.info(f"Starting Phoonnx voice selection (force={force}).")
        
        try:
            voices = get_remote_voices_for_gui()

            if not voices:
                # "No voices found on the server."
                wx.CallAfter(gui.messageBox, "No voices found on the server.", "Error")
                return

            saved_id, saved_ver = get_selected_voice()
            log.info(f"Current voice in NVDA: {saved_id} (version {saved_ver})")

            if saved_id:
                current = next((v for v in voices if v.get("id") == saved_id), None)
                
                if current and not force and current.get("version") == saved_ver:
                    log.info("Voice is up to date. No download needed.")
                    return
                elif current:
                    def prompt_update():
                        # "A new version of '{current['name']}' is available (or you forced the selection).\n" \
                        # "Do you want to download and set it up again now?"
                        msg = f"A new version of '{current['name']}' is available (or you forced the selection).\n" \
                              f"Do you want to download and set it up again now?"
                        # "Update/Selection Available"
                        if gui.messageBox(msg, "Update/Selection Available", style=wx.YES_NO) == wx.YES:
                            # Download in the worker thread
                            threading.Thread(target=download_voice, args=(current,), daemon=True).start()
                    wx.CallAfter(prompt_update)
                    return
            
            # Execute the selection dialog in the GUI thread
            suggestion = suggest_voice(voices)
            
            def show_dialog_and_download():
                selected = show_voice_selection_dialog(voices, suggestion)
                if selected:
                    # Download in the worker thread
                    threading.Thread(target=download_voice, args=(selected,), daemon=True).start()

            wx.CallAfter(show_dialog_and_download)

        except Exception as e:
            log.error(f"Error during voice selection: {e}", exc_info=True)
            wx.CallAfter(gui.messageBox, 
                         # "An unexpected error occurred during voice selection:\n{e}"
                         f"An unexpected error occurred during voice selection:\n{e}", 
                         "Error", 
                         style=wx.ICON_ERROR)

    # Start the logic in a separate thread
    threading.Thread(target=run_in_thread, daemon=True).start()