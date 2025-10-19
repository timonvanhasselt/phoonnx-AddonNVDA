# globalPlugins/phoonnx/phoonnxSettingsPanel.py

import os
import sys
import wx
import threading
from logHandler import log
import globalPluginHandler
import ui 
from scriptHandler import script
import api 
import synthDriverHandler 
from gui.settingsDialogs import SettingsPanel 
import gui 
from pathlib import Path 
import json 
import shutil # Needed for moving and deleting directories

# Import the translation function
_ = lambda s: s 

# --- Path configuration (Crucial for importing all bundled libs) ---

# The base directory of the add-on: .../Phoonnx TTS Driver 64 bit
ADDON_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# The directory containing all root packages (ovos_plugin_manager, phoonnx, etc.)
PHOONNX_LIBS_PATH = os.path.join(ADDON_ROOT_DIR, 'phoonnx_libs')

# The **ONLY** directory where we want to store/check voices (Destination path).
VOICE_INSTALL_DIR = Path(os.path.join(ADDON_ROOT_DIR, 'synthDrivers', 'phoonnx', 'voices'))

# The file path for the manager state
VOICE_MANAGER_STATE_FILE = VOICE_INSTALL_DIR.parent / "voices_cache.json"

# Helper function to get the installation path (Destination path)
def _get_addon_voice_path(voice_id):
    """Returns the path to the voice installation directory within the add-on."""
    return VOICE_INSTALL_DIR / voice_id

# NEW: Helper function for the hardcoded download location (Source path)
def _get_hardcoded_cache_path(voice_id):
    """Calculates the hardcoded cache location where the voice downloads (must match model_manager.py)."""
    # This path is hardcoded in TTSModelInfo.voice_path in the Phoonnx library
    return Path(os.path.expanduser("~")) / ".cache" / "phoonnx" / "voices" / voice_id


# ADD THIS PATH TO sys.path SO Python CAN FIND THE BUNDLED DEPENDENCIES
if PHOONNX_LIBS_PATH not in sys.path:
    sys.path.insert(0, PHOONNX_LIBS_PATH)
    log.debug(f"Phoonnx Voice Manager: Added libs path: {PHOONNX_LIBS_PATH}")


# --- Function to load the Voice Manager ---
def get_model_manager_and_voices():
    """Loads the TTSModelManager and retrieves a list of voice information."""
    
    TTSModelManager = None
    TTSModelInfo = None
    
    try:
        from phoonnx.model_manager import TTSModelManager, TTSModelInfo
        
    except ImportError as e:
        log.error(f"FATAL ERROR: Cannot import Phoonnx modules: {e}", exc_info=True)
        return None, [], None, None
    except Exception as e:
        log.error(f"FATAL ERROR: Unknown error during import: {e}", exc_info=True)
        return None, [], None, None

    # Direct instance of the manager
    try:
        # Initialize the manager without model_dir, as it is hardcoded in model_manager.py
        manager = TTSModelManager(cache_path=str(VOICE_MANAGER_STATE_FILE)) 
        
        # Ensure the voices directory exists
        VOICE_INSTALL_DIR.mkdir(parents=True, exist_ok=True)
        log.debug(f"Phoonnx Manager uses cache file: {VOICE_MANAGER_STATE_FILE}")

        return manager, [], TTSModelManager, TTSModelInfo
        
    except Exception as e:
        log.error(f"Error initializing Phoonnx Model Manager: {e}", exc_info=True)
        return None, [], None, None


# NEW: Helper function for status check
def is_voice_installed(info):
    """Checks if a voice is actually installed locally (model file exists)."""
    
    # We check the add-on directory, NOT the cache directory
    model_dir = _get_addon_voice_path(info.voice_id)
    
    if not model_dir.is_dir():
        return False
        
    # Search for ONNX or PT files (the actual models)
    model_files = list(model_dir.glob("*.onnx")) + list(model_dir.glob("*.pt"))
    
    return len(model_files) > 0


# --- The Settings Panel (GUI) ---
class PhoonnxVoiceManagerPanel(SettingsPanel):
    """Settings panel for managing Phoonnx voices (downloading, selecting)."""
    
    title = _("Phoonnx voices")

    def makeSettings(self, settingsSizer):
        """Builds the user interface of the panel."""
        sHelper = gui.guiHelper.BoxSizerHelper(self, sizer=settingsSizer)
        
        # The manager is initialized without forcing the download directory (because we are going to move)
        self.manager, self.voices, self.TTSModelManager, self.TTSModelInfo = get_model_manager_and_voices()
        
        self.loading_label = None
        self.sList = None
        self.sDetails = None
        self.sDownloadButton = None
        self.sDeleteButton = None
        # self.sSetDefaultButton = None # REMOVED
        
        self.buttonSizer = None 
        
        if not self.manager:
            error_label = wx.StaticText(self, wx.ID_ANY, _("Error: Cannot load Phoonnx Voice Manager. See NVDA log for details."))
            sHelper.addItem(error_label, proportion=0, flag=wx.ALL, border=10)
        else:
            self.loading_label = wx.StaticText(self, wx.ID_ANY, _("Loading voices list (may take a moment)..."))
            sHelper.addItem(self.loading_label, proportion=0, flag=wx.ALL | wx.ALIGN_CENTER, border=10)
            
            self._add_hidden_controls(sHelper)
            
            self._load_voices_async()
            
            self.update_button_states()

    def _add_hidden_controls(self, sHelper):
        """Adds the actual voice management elements (hidden by default)."""
        
        self.available_label = wx.StaticText(self, wx.ID_ANY, _("Available voices:"))
        sHelper.addItem(self.available_label, proportion=0, flag=wx.ALL | wx.ALIGN_LEFT, border=5)
        self.available_label.Hide()

        self.sList = wx.ListBox(self, wx.ID_ANY, choices=[])
        sHelper.addItem(self.sList, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
        self.sList.SetToolTip(_("List of all available voices."))
        self.sList.Hide()
        
        self.sDetails = wx.StaticText(self, wx.ID_ANY, _("Select a voice for details..."))
        sHelper.addItem(self.sDetails, proportion=0, flag=wx.EXPAND | wx.ALL, border=5)
        self.sDetails.Hide()
        
        self.buttonSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.sDownloadButton = wx.Button(self, wx.ID_ANY, _("Download/Update"))
        self.sDeleteButton = wx.Button(self, wx.ID_ANY, _("Delete"))
        # self.sSetDefaultButton is REMOVED
        
        self.buttonSizer.Add(self.sDownloadButton, 0, wx.ALL, 5)
        self.buttonSizer.Add(self.sDeleteButton, 0, wx.ALL, 5)
        # self.buttonSizer.Add(self.sSetDefaultButton, 0, wx.ALL, 5) # REMOVED
        
        sHelper.addItem(self.buttonSizer, proportion=0, flag=wx.ALIGN_CENTER | wx.ALL, border=5)
        
        self.buttonSizer.Show(False)

        self.sList.Bind(wx.EVT_LISTBOX, self.onVoiceSelect)
        self.sDownloadButton.Bind(wx.EVT_BUTTON, self.onDownload)
        self.sDeleteButton.Bind(wx.EVT_BUTTON, self.onDelete)
        # self.sSetDefaultButton.Bind(wx.EVT_BUTTON, self.onSetDefault) # REMOVED

    def _load_voices_async(self):
        """Starts a thread to retrieve the voice list."""
        
        def run_load():
            try:
                # The manager will now check the configured VOICE_INSTALL_DIR.
                self.manager.refresh_voices()
            except Exception as exc:
                log.warning(f"Voice refresh failed; attempting load: {exc}")
                self.manager.load()
                
            self.voices = list(self.manager.voices.values())
            
            wx.CallAfter(self.on_voices_loaded)

        if self.manager:
            threading.Thread(target=run_load, daemon=True).start()

    def on_voices_loaded(self):
        """Called on the MainThread after the voice list has been loaded."""
        if not self.manager:
            return 
            
        # 1. Hide the 'Loading...' message
        if self.loading_label:
            self.loading_label.Hide()
            
        # 2. CURRENT VOICES: Create map of manager's voices (IDs)
        manager_voices_ids = {v.voice_id for v in self.voices}
        
        # 3. CHECK FOR LOCALLY INSTALLED, UNKNOWN VOICES
        log.info(f"Phoonnx: Checking for locally installed voices in {VOICE_INSTALL_DIR}")
        
        # Loop through the voice directories at the installation location
        for voice_dir in VOICE_INSTALL_DIR.iterdir():
            if voice_dir.is_dir():
                voice_id = voice_dir.name
                
                # Check 1: Is there a model file?
                has_model_file = len(list(voice_dir.glob("*.onnx")) + list(voice_dir.glob("*.pt"))) > 0
                
                # Check 2: Is the voice already known to the manager AND is it locally installed?
                if voice_id not in manager_voices_ids and has_model_file:
                    
                    log.warning(f"Phoonnx: Local voice '{voice_id}' is unknown to the manager. Adding to the list.")
                    
                    # Try to parse language
                    lang_tag = voice_id.split('_')[0] if '_' in voice_id else 'unk'
                    
                    # Try to load local config (for a better TTSModelInfo)
                    config_available = False
                    config_path = voice_dir / "model.json"
                    if config_path.exists():
                        config_available = True 
                    
                    try:
                        # Create a TTSModelInfo object 
                        local_info = self.TTSModelInfo(
                            voice_id=voice_id,
                            lang=lang_tag,
                            model_url="local", # Dummy URL
                            config_url="local", # Dummy URL
                            config=config_available # Mark as locally configured/present
                        )
                        self.voices.append(local_info)
                        log.info(f"Phoonnx: Local voice {voice_id} successfully added to the manager list.")
                    except Exception as e:
                        log.error(f"Error creating TTSModelInfo for local voice {voice_id}: {e}", exc_info=True)


        # 4. Update the ListBox with the results (including locally added voices)
        voice_names = []
        for info in self.voices:
            is_installed = is_voice_installed(info) 
            status = _("(Installed)") if is_installed else _("(Not installed)")
            voice_names.append(f"{info.voice_id} ({info.lang}) - {status}")
            
        self.sList.Set(voice_names)
        
        # 5. Show the elements
        self.available_label.Show()
        self.sList.Show()
        self.sDetails.Show()
        self.buttonSizer.Show(True)
        self.GetSizer().Layout()
        
        self.Layout()
        self.GetParent().Layout()
        
        # 6. Update the button status
        self.update_button_states()

    def update_button_states(self, selected_voice_info=None):
        """Enables/disables the buttons based on the selected voice status."""
        if self.sDownloadButton is None:
            return 
            
        if selected_voice_info:
            is_installed = is_voice_installed(selected_voice_info)
            self.sDownloadButton.Enable(True)
            self.sDeleteButton.Enable(is_installed)
            # self.sSetDefaultButton enablement logic REMOVED
        else:
            self.sDownloadButton.Enable(False)
            self.sDeleteButton.Enable(False)
            # self.sSetDefaultButton enablement logic REMOVED

    def onVoiceSelect(self, evt):
        selected_index = self.sList.GetSelection()
        if selected_index == wx.NOT_FOUND or not self.voices: 
            self.update_button_states()
            return

        selected_voice_info = self.voices[selected_index]
        is_installed = is_voice_installed(selected_voice_info)

        details = (
            _("Name: ") + selected_voice_info.voice_id + "\n"
            + _("ID: ") + selected_voice_info.voice_id + "\n"
            + _("Language: ") + selected_voice_info.lang + "\n"
            + _("Installed: ") + (_("Yes") if is_installed else _("No"))
        )
        self.sDetails.SetLabel(details)
        self.update_button_states(selected_voice_info)

    # --- Helper functions for Download/Delete ---
    
    def _get_voice_info(self, voice_id):
        """Looks up the TTSModelInfo based on the ID."""
        voice_info = next((v for v in self.voices if v.voice_id == voice_id), None)
        if not voice_info:
            raise ValueError(f"Voice '{voice_id}' not found in the list.")
        return voice_info

    def _download_voice(self, voice_id):
        """Wrapper for the download task. Loads the voice and then moves it to the add-on directory."""
        
        # 1. Start the download to the hardcoded cache directory (~/.cache/phoonnx/voices/)
        voice_info = self._get_voice_info(voice_id)
        # The download happens here (and internally uses the hardcoded cache directory)
        voice_info.load() 

        # 2. Determine the paths
        source_path = _get_hardcoded_cache_path(voice_id) # The hardcoded cache directory (Source)
        dest_path = _get_addon_voice_path(voice_id)     # The NVDA add-on directory (Destination)

        if source_path.is_dir():
            # 3. Move from the cache to the NVDA add-on directory
            if dest_path.exists():
                # Delete the old directory in the add-on for a clean move (for Update)
                log.info(f"Phoonnx: Removing existing add-on voice directory for update: {dest_path}")
                shutil.rmtree(dest_path) 
            
            # Recursively move the directory from the cache to the add-on directory
            shutil.move(str(source_path), str(dest_path)) 
            log.info(f"Phoonnx: Files for {voice_id} successfully moved from cache ({source_path}) to add-on directory ({dest_path}).")
            
            # OPTIONAL: Try to delete the empty parent cache directory, if empty
            try:
                source_parent = source_path.parent
                if source_parent.exists() and not list(source_parent.iterdir()):
                    source_parent.rmdir()
            except OSError:
                pass
        else:
            log.error(f"Phoonnx: Download complete, but the source directory {source_path} was not found. Cannot move.")


    def _delete_voice(self, voice_id):
        """Wrapper for the delete task. Deletes the directory manually."""
        voice_path = _get_addon_voice_path(voice_id)
        if voice_path.exists():
            log.info(f"Phoonnx: Manually deleting files: {voice_path}")
            # Recursively delete the directory
            shutil.rmtree(voice_path)


    # --- Background Tasks (Wrapper) ---

    def _execute_async_task(self, task_func, success_msg, voice_id, refresh_synthesizer=False):
        """Executes a task in a separate thread for download/deletion."""
        
        self.sDownloadButton.Disable()
        self.sDeleteButton.Disable()
        
        # Use ui.message() instead of api.speak()
        ui.message(_(f"Starting operation for voice {voice_id}.")) 
        
        def wrapper():
            try:
                task_func(voice_id)
                
                if refresh_synthesizer:
                    # **CORRECTION 2: Replacement of the outdated/insufficient refresh call.**
                    current_synth = synthDriverHandler.getSynth()
                    if current_synth and current_synth.name == "phoonnx":
                        # Resetting the synthesizer forces a reload of the voice list.
                        synthDriverHandler.setSynth(current_synth.name)
                        log.info(f"Phoonnx: Synthesizer reloaded to update the voice list after operation for {voice_id}.")
                    else:
                         log.warning("Phoonnx: Cannot reload Phoonnx synthesizer (not the current active one).")

                    
                wx.CallAfter(lambda: self.on_async_task_complete(True, success_msg))
            except Exception as e:
                log.error(f"Error during asynchronous task for voice {voice_id}: {e}", exc_info=True)
                # Give a shorter spoken message
                wx.CallAfter(lambda: self.on_async_task_complete(False, _("Error: Operation failed. See NVDA log.")))

        threading.Thread(target=wrapper, daemon=True).start()


    def on_async_task_complete(self, success, message):
        """Called after an asynchronous task is completed."""
        
        # Reload the voices to update the status in the ListBox
        self.manager.refresh_voices()
        self.voices = list(self.manager.voices.values())
        
        voice_names = []
        for info in self.voices:
            status = _("(Installed)") if is_voice_installed(info) else _("(Not installed)")
            voice_names.append(f"{info.voice_id} ({info.lang}) - {status}")
        self.sList.Set(voice_names)
        
        self.onVoiceSelect(None) 
        
        # Use ui.message() for spoken feedback
        ui.message(message) 
        
        dlg = wx.MessageDialog(self, message, 
                               _("Operation complete") if success else _("Operation failed"), 
                               wx.OK | (wx.ICON_INFORMATION if success else wx.ICON_ERROR))
        dlg.ShowModal()
        dlg.Destroy()
        
        self.onVoiceSelect(None)
        

    # --- Button Handlers ---

    def onDownload(self, evt):
        selected_index = self.sList.GetSelection()
        if selected_index == wx.NOT_FOUND: return
        voice_info = self.voices[selected_index]
        
        self._execute_async_task(
            task_func=self._download_voice, 
            success_msg=_("Download is complete. The voice is now available in the synthesizer settings."),
            voice_id=voice_info.voice_id,
            refresh_synthesizer=True
        )

    def onDelete(self, evt):
        selected_index = self.sList.GetSelection()
        if selected_index == wx.NOT_FOUND: return
        voice_info = self.voices[selected_index]
        
        self._execute_async_task(
            task_func=self._delete_voice, 
            success_msg=_("The voice has been successfully deleted. The synthesizer voice list has been updated."),
            voice_id=voice_info.voice_id,
            refresh_synthesizer=True
        )

    # def onSetDefault(self, evt): # REMOVED (Formerly 18 lines of code removed)

    def onSave(self):
        """Required method for SettingsPanel. Actions are executed immediately."""
        pass
        
    @script(gesture="kb:NVDA+shift+v") 
    def script_showVoiceManagerPanel(self, gesture):
        """Shows the NVDA settings window, focused on the Phoonnx panel."""
        gui.mainFrame.runSettingsDialog(gui.settingsDialogs.NVDASettingsDialog,
                                        startCategory=PhoonnxVoiceManagerPanel)
