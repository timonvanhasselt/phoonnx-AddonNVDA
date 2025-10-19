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
    # Cruciale import van de klasse uit het .py bestand in dezelfde map
    from .phoonnxSettingsPanel import PhoonnxVoiceManagerPanel
    log.info("Phoonnx Global Plugin: SUCCESS: PhoonnxVoiceManagerPanel succesvol geïmporteerd.")
except ImportError as e:
    # Dit zal de fout loggen als de import mislukt
    log.critical(f"FATALE FOUT: Kan PhoonnxVoiceManagerPanel NIET importeren uit phoonnxSettingsPanel. Paneel zal niet verschijnen. Fout: {e}", exc_info=True)


class GlobalPlugin(globalPluginHandler.GlobalPlugin):
    """
    De hoofdklasse die NVDA laadt en die het instellingenpaneel registreert.
    """
    def __init__(self):
        super().__init__()
        # De registratie van het paneel in NVDA's instellingen
        if PhoonnxVoiceManagerPanel:
            gui.settingsDialogs.NVDASettingsDialog.categoryClasses.append(PhoonnxVoiceManagerPanel)
            log.info("Phoonnx Global Plugin: Settings Panel succesvol GEREGISTREERD in NVDA.")
        else:
            log.warning("Phoonnx Global Plugin: Settings Panel registratie OVERGESLAGEN (import mislukt).")

    def terminate(self):
        super().terminate()
        # Zorg ervoor dat het paneel wordt verwijderd bij het afsluiten
        if PhoonnxVoiceManagerPanel and PhoonnxVoiceManagerPanel in gui.settingsDialogs.NVDASettingsDialog.categoryClasses:
            gui.settingsDialogs.NVDASettingsDialog.categoryClasses.remove(PhoonnxVoiceManagerPanel)