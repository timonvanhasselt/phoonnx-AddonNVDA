# globalPlugins/phoonnx/globalPlugins.py
import globalPluginHandler
import gui
import addonHandler
from logHandler import log
import sys
import os

addonHandler.initTranslation()

PhoonnxVoiceManagerPanel = None
try:
    # Crucial import of the class from the .py file in the same directory
    from .phoonnxSettingsPanel import PhoonnxVoiceManagerPanel
    log.info("Phoonnx Global Plugin: SUCCESS: PhoonnxVoiceManagerPanel successfully imported.")
except ImportError as e:
    # This will log the error if the import fails
    log.critical(f"FATAL ERROR: Cannot import PhoonnxVoiceManagerPanel from phoonnxSettingsPanel. Panel will not appear. Error: {e}", exc_info=True)


class GlobalPlugin(globalPluginHandler.GlobalPlugin):
    """
    The main class that NVDA loads and which registers the settings panel.
    """
    def __init__(self):
        super().__init__()
        # Registration of the panel in NVDA's settings
        if PhoonnxVoiceManagerPanel:
            gui.settingsDialogs.NVDASettingsDialog.categoryClasses.append(PhoonnxVoiceManagerPanel)
            log.info("Phoonnx Global Plugin: Settings Panel successfully REGISTERED in NVDA.")
        else:
            log.warning("Phoonnx Global Plugin: Settings Panel registration SKIPPED (import failed).")

    def terminate(self):
        super().terminate()
        # Ensure the panel is removed upon termination
        if PhoonnxVoiceManagerPanel and PhoonnxVoiceManagerPanel in gui.settingsDialogs.NVDASettingsDialog.categoryClasses:
            gui.settingsDialogs.NVDASettingsDialog.categoryClasses.remove(PhoonnxVoiceManagerPanel)
