# globalPlugins/phoonnx/phoonnxSettingsPanel.py

import os
import sys
import wx
import threading
# import config  <-- VERWIJDERD
from logHandler import log
import globalPluginHandler
import ui 
from scriptHandler import script
import api 
# **CORRECTIE 1: Nodig voor het vernieuwen van de stemmenlijst**
import synthDriverHandler 
from gui.settingsDialogs import SettingsPanel 
import gui 
from pathlib import Path 
import json 
import shutil # Nodig voor het verplaatsen en verwijderen van mappen

# Importeer de vertaalfunctie
_ = lambda s: s 

# --- Pad configuratie (Cruciaal voor het importeren van alle gebundelde libs) ---

# De basismap van de add-on: .../Phoonnx TTS Driver 64 bit
ADDON_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# De map die alle root-pakketten bevat (ovos_plugin_manager, phoonnx, etc.)
PHOONNX_LIBS_PATH = os.path.join(ADDON_ROOT_DIR, 'phoonnx_libs')

# De **ENIGE** map waar we stemmen willen opslaan/controleren (Bestemmingspad).
VOICE_INSTALL_DIR = Path(os.path.join(ADDON_ROOT_DIR, 'synthDrivers', 'phoonnx', 'voices'))

# Het bestandspad voor de manager state
VOICE_MANAGER_STATE_FILE = VOICE_INSTALL_DIR.parent / "voices_cache.json"

# Hulpfunctie om het installatiepad te krijgen (Bestemmingspad)
def _get_addon_voice_path(voice_id):
    """Retourneert het pad naar de installatiemap van de stem binnen de add-on."""
    return VOICE_INSTALL_DIR / voice_id

# NIEUW: Hulpfunctie voor de hardcoded downloadlocatie (Bronpad)
def _get_hardcoded_cache_path(voice_id):
    """Berekent de hardcoded cache locatie waar de stem downloadt (moet overeenkomen met model_manager.py)."""
    # Dit pad is hardcoded in TTSModelInfo.voice_path in de Phoonnx bibliotheek
    return Path(os.path.expanduser("~")) / ".cache" / "phoonnx" / "voices" / voice_id


# VOEG DIT PAD TOE AAN sys.path ZODAT Python DE GEBUNDELDE AFHANKELIJKHEDEN KAN VINDEN
if PHOONNX_LIBS_PATH not in sys.path:
    sys.path.insert(0, PHOONNX_LIBS_PATH)
    log.debug(f"Phoonnx Voice Manager: Toegevoegd libs pad: {PHOONNX_LIBS_PATH}")


# --- Functie om de Voice Manager te laden ---
def get_model_manager_and_voices():
    """Laadt de TTSModelManager en haalt een lijst met steminformatie op."""
    
    TTSModelManager = None
    TTSModelInfo = None
    
    try:
        from phoonnx.model_manager import TTSModelManager, TTSModelInfo
        
    except ImportError as e:
        log.error(f"FATALE FOUT: Kan Phoonnx modules niet importeren: {e}", exc_info=True)
        return None, [], None, None
    except Exception as e:
        log.error(f"FATALE FOUT: Onbekende fout tijdens import: {e}", exc_info=True)
        return None, [], None, None

    # Directe instantie van de manager
    try:
        # Initialiseer de manager zonder model_dir, aangezien deze hardcoded is in model_manager.py
        manager = TTSModelManager(cache_path=str(VOICE_MANAGER_STATE_FILE)) 
        
        # Zorg ervoor dat de stemmen-map bestaat
        VOICE_INSTALL_DIR.mkdir(parents=True, exist_ok=True)
        log.debug(f"Phoonnx Manager gebruikt cache file: {VOICE_MANAGER_STATE_FILE}")

        return manager, [], TTSModelManager, TTSModelInfo
        
    except Exception as e:
        log.error(f"Fout bij het initialiseren van Phoonnx Model Manager: {e}", exc_info=True)
        return None, [], None, None


# NIEUW: Hulpfunctie voor statuscontrole
def is_voice_installed(info):
    """Controleert of een stem daadwerkelijk lokaal is geïnstalleerd (modelbestand bestaat)."""
    
    # We controleren de add-on map, NIET de cache map
    model_dir = _get_addon_voice_path(info.voice_id)
    
    if not model_dir.is_dir():
        return False
        
    # Zoek naar ONNX of PT bestanden (de feitelijke modellen)
    model_files = list(model_dir.glob("*.onnx")) + list(model_dir.glob("*.pt"))
    
    return len(model_files) > 0


# --- Het Instellingenpaneel (GUI) ---
class PhoonnxVoiceManagerPanel(SettingsPanel):
    """Instellingenpaneel voor het beheren van Phoonnx-stemmen (downloaden, selecteren)."""
    
    title = _("Phoonnx Stemmenbeheer")

    def makeSettings(self, settingsSizer):
        """Bouwt de gebruikersinterface van het paneel."""
        sHelper = gui.guiHelper.BoxSizerHelper(self, sizer=settingsSizer)
        
        # De manager wordt geïnitialiseerd zonder de downloadmap te forceren (omdat we gaan verplaatsen)
        self.manager, self.voices, self.TTSModelManager, self.TTSModelInfo = get_model_manager_and_voices()
        
        self.loading_label = None
        self.sList = None
        self.sDetails = None
        self.sDownloadButton = None
        self.sDeleteButton = None
        # self.sSetDefaultButton = None # VERWIJDERD
        
        self.buttonSizer = None 
        
        if not self.manager:
            error_label = wx.StaticText(self, wx.ID_ANY, _("Fout: Kan Phoonnx Voice Manager niet laden. Zie NVDA-log voor details."))
            sHelper.addItem(error_label, proportion=0, flag=wx.ALL, border=10)
        else:
            self.loading_label = wx.StaticText(self, wx.ID_ANY, _("Laden van stemmenlijst (kan even duren)..."))
            sHelper.addItem(self.loading_label, proportion=0, flag=wx.ALL | wx.ALIGN_CENTER, border=10)
            
            self._add_hidden_controls(sHelper)
            
            self._load_voices_async()
            
            self.update_button_states()

    def _add_hidden_controls(self, sHelper):
        """Voegt de daadwerkelijke stemmenbeheer elementen toe (standaard verborgen)."""
        
        self.available_label = wx.StaticText(self, wx.ID_ANY, _("Beschikbare stemmen:"))
        sHelper.addItem(self.available_label, proportion=0, flag=wx.ALL | wx.ALIGN_LEFT, border=5)
        self.available_label.Hide()

        self.sList = wx.ListBox(self, wx.ID_ANY, choices=[])
        sHelper.addItem(self.sList, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
        self.sList.SetToolTip(_("Lijst van alle beschikbare stemmen."))
        self.sList.Hide()
        
        self.sDetails = wx.StaticText(self, wx.ID_ANY, _("Selecteer een stem voor details..."))
        sHelper.addItem(self.sDetails, proportion=0, flag=wx.EXPAND | wx.ALL, border=5)
        self.sDetails.Hide()
        
        self.buttonSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.sDownloadButton = wx.Button(self, wx.ID_ANY, _("Download/Update"))
        self.sDeleteButton = wx.Button(self, wx.ID_ANY, _("Verwijderen"))
        # self.sSetDefaultButton is VERWIJDERD
        
        self.buttonSizer.Add(self.sDownloadButton, 0, wx.ALL, 5)
        self.buttonSizer.Add(self.sDeleteButton, 0, wx.ALL, 5)
        # self.buttonSizer.Add(self.sSetDefaultButton, 0, wx.ALL, 5) # VERWIJDERD
        
        sHelper.addItem(self.buttonSizer, proportion=0, flag=wx.ALIGN_CENTER | wx.ALL, border=5)
        
        self.buttonSizer.Show(False)

        self.sList.Bind(wx.EVT_LISTBOX, self.onVoiceSelect)
        self.sDownloadButton.Bind(wx.EVT_BUTTON, self.onDownload)
        self.sDeleteButton.Bind(wx.EVT_BUTTON, self.onDelete)
        # self.sSetDefaultButton.Bind(wx.EVT_BUTTON, self.onSetDefault) # VERWIJDERD

    def _load_voices_async(self):
        """Start een thread om de stemmenlijst op te halen."""
        
        def run_load():
            try:
                # De manager zal nu de geconfigureerde VOICE_INSTALL_DIR controleren.
                self.manager.refresh_voices()
            except Exception as exc:
                log.warning(f"Voice refresh failed; attempting load: {exc}")
                self.manager.load()
                
            self.voices = list(self.manager.voices.values())
            
            wx.CallAfter(self.on_voices_loaded)

        if self.manager:
            threading.Thread(target=run_load, daemon=True).start()

    def on_voices_loaded(self):
        """Wordt op de MainThread aangeroepen nadat de stemmenlijst is geladen."""
        if not self.manager:
            return 
            
        # 1. Verberg het 'Laden...' bericht
        if self.loading_label:
            self.loading_label.Hide()
            
        # 2. HUIDIGE STEMMEN: Map maken van manager's stemmen (ID's)
        manager_voices_ids = {v.voice_id for v in self.voices}
        
        # 3. CONTROLEER OP LOKAAL GEÏNSTALLEERDE, ONBEKENDE STEMMEN
        log.info(f"Phoonnx: Controle op lokaal geïnstalleerde stemmen in {VOICE_INSTALL_DIR}")
        
        # Loop door de stem-mappen op de installatielocatie
        for voice_dir in VOICE_INSTALL_DIR.iterdir():
            if voice_dir.is_dir():
                voice_id = voice_dir.name
                
                # Check 1: Is er een modelbestand?
                has_model_file = len(list(voice_dir.glob("*.onnx")) + list(voice_dir.glob("*.pt"))) > 0
                
                # Check 2: Is de stem al bekend bij de manager EN is het lokaal geïnstalleerd?
                if voice_id not in manager_voices_ids and has_model_file:
                    
                    log.warning(f"Phoonnx: Lokale stem '{voice_id}' is niet bekend bij de manager. Voeg toe aan de lijst.")
                    
                    # Probeer taal te parsen
                    lang_tag = voice_id.split('_')[0] if '_' in voice_id else 'unk'
                    
                    # Probeer lokale config te laden (voor een betere TTSModelInfo)
                    config_available = False
                    config_path = voice_dir / "model.json"
                    if config_path.exists():
                        config_available = True 
                    
                    try:
                        # Maak een TTSModelInfo object 
                        local_info = self.TTSModelInfo(
                            voice_id=voice_id,
                            lang=lang_tag,
                            model_url="local", # Dummy URL
                            config_url="local", # Dummy URL
                            config=config_available # Markeer als lokaal geconfigureerd/aanwezig
                        )
                        self.voices.append(local_info)
                        log.info(f"Phoonnx: Lokale stem {voice_id} succesvol toegevoegd aan de manager lijst.")
                    except Exception as e:
                        log.error(f"Fout bij het creëren van TTSModelInfo voor lokale stem {voice_id}: {e}", exc_info=True)


        # 4. Update de ListBox met de resultaten (inclusief lokaal toegevoegde stemmen)
        voice_names = []
        for info in self.voices:
            is_installed = is_voice_installed(info) 
            status = _("(Geïnstalleerd)") if is_installed else _("(Niet geïnstalleerd)")
            voice_names.append(f"{info.voice_id} ({info.lang}) - {status}")
            
        self.sList.Set(voice_names)
        
        # 5. Toon de elementen
        self.available_label.Show()
        self.sList.Show()
        self.sDetails.Show()
        self.buttonSizer.Show(True)
        self.GetSizer().Layout()
        
        self.Layout()
        self.GetParent().Layout()
        
        # 6. Werk de knopstatus bij
        self.update_button_states()

    def update_button_states(self, selected_voice_info=None):
        """Schakelt de knoppen in/uit op basis van de geselecteerde stemstatus."""
        if self.sDownloadButton is None:
            return 
            
        if selected_voice_info:
            is_installed = is_voice_installed(selected_voice_info)
            self.sDownloadButton.Enable(True)
            self.sDeleteButton.Enable(is_installed)
            # self.sSetDefaultButton enablement logic VERWIJDERD
        else:
            self.sDownloadButton.Enable(False)
            self.sDeleteButton.Enable(False)
            # self.sSetDefaultButton enablement logic VERWIJDERD

    def onVoiceSelect(self, evt):
        selected_index = self.sList.GetSelection()
        if selected_index == wx.NOT_FOUND or not self.voices: 
            self.update_button_states()
            return

        selected_voice_info = self.voices[selected_index]
        is_installed = is_voice_installed(selected_voice_info)

        details = (
            _("Naam: ") + selected_voice_info.voice_id + "\n"
            + _("ID: ") + selected_voice_info.voice_id + "\n"
            + _("Taal: ") + selected_voice_info.lang + "\n"
            + _("Geïnstalleerd: ") + (_("Ja") if is_installed else _("Nee"))
        )
        self.sDetails.SetLabel(details)
        self.update_button_states(selected_voice_info)

    # --- Hulpfuncties voor Download/Delete ---
    
    def _get_voice_info(self, voice_id):
        """Zoekt de TTSModelInfo op basis van de ID."""
        voice_info = next((v for v in self.voices if v.voice_id == voice_id), None)
        if not voice_info:
            raise ValueError(f"Stem '{voice_id}' niet gevonden in de lijst.")
        return voice_info

    def _download_voice(self, voice_id):
        """Wrapper voor de download taak. Laadt de stem en verplaatst deze daarna naar de add-on map."""
        
        # 1. Start de download naar de hardcoded cache map (~/.cache/phoonnx/voices/)
        voice_info = self._get_voice_info(voice_id)
        # De download gebeurt hier (en gebruikt intern de hardcoded cache map)
        voice_info.load() 

        # 2. Bepaal de paden
        source_path = _get_hardcoded_cache_path(voice_id) # De hardcoded cache map (Bron)
        dest_path = _get_addon_voice_path(voice_id)     # De NVDA add-on map (Bestemming)

        if source_path.is_dir():
            # 3. Verplaatsen van de cache naar de NVDA add-on map
            if dest_path.exists():
                # Verwijder de oude map in de add-on voor een schone verplaatsing (bij Update)
                log.info(f"Phoonnx: Verwijder bestaande add-on stemmap voor update: {dest_path}")
                shutil.rmtree(dest_path) 
            
            # Verplaats de map recursief van de cache naar de add-on map
            shutil.move(str(source_path), str(dest_path)) 
            log.info(f"Phoonnx: Bestanden voor {voice_id} succesvol verplaatst van cache ({source_path}) naar add-on map ({dest_path}).")
            
            # OPTIONEEL: Probeer de lege parent cache map te verwijderen, indien leeg
            try:
                source_parent = source_path.parent
                if source_parent.exists() and not list(source_parent.iterdir()):
                    source_parent.rmdir()
            except OSError:
                pass
        else:
            log.error(f"Phoonnx: Download voltooid, maar de bronmap {source_path} is niet gevonden. Kan niet verplaatsen.")


    def _delete_voice(self, voice_id):
        """Wrapper voor de verwijder taak. Verwijder de map handmatig."""
        voice_path = _get_addon_voice_path(voice_id)
        if voice_path.exists():
            log.info(f"Phoonnx: Bestanden handmatig verwijderen: {voice_path}")
            # Verwijder de map recursief
            shutil.rmtree(voice_path)


    # --- Achtergrondtaken (Wrapper) ---

    def _execute_async_task(self, task_func, success_msg, voice_id, refresh_synthesizer=False):
        """Voert een taak uit in een aparte thread voor download/verwijdering."""
        
        self.sDownloadButton.Disable()
        self.sDeleteButton.Disable()
        
        # Gebruik ui.message() in plaats van api.speak()
        ui.message(_(f"Start de operatie voor stem {voice_id}.")) 
        
        def wrapper():
            try:
                task_func(voice_id)
                
                if refresh_synthesizer:
                    # **CORRECTIE 2: Vervanging van de verouderde/onvoldoende refresh-aanroep.**
                    current_synth = synthDriverHandler.getSynth()
                    if current_synth and current_synth.name == "phoonnx":
                        # De synthesizer opnieuw instellen dwingt een herlading van de stemmenlijst af.
                        synthDriverHandler.setSynth(current_synth.name)
                        log.info(f"Phoonnx: Synthesizer is opnieuw geladen om de stemmenlijst bij te werken na operatie van {voice_id}.")
                    else:
                         log.warning("Phoonnx: Kan Phoonnx synthesizer niet opnieuw laden (niet de huidige actieve).")

                    
                wx.CallAfter(lambda: self.on_async_task_complete(True, success_msg))
            except Exception as e:
                log.error(f"Fout bij asynchrone taak voor stem {voice_id}: {e}", exc_info=True)
                # Geef een kortere gesproken melding
                wx.CallAfter(lambda: self.on_async_task_complete(False, _("Fout: De operatie is mislukt. Zie NVDA-log.")))

        threading.Thread(target=wrapper, daemon=True).start()


    def on_async_task_complete(self, success, message):
        """Wordt aangeroepen nadat een asynchrone taak is voltooid."""
        
        # Laad de stemmen opnieuw om de status in de ListBox bij te werken
        self.manager.refresh_voices()
        self.voices = list(self.manager.voices.values())
        
        voice_names = []
        for info in self.voices:
            status = _("(Geïnstalleerd)") if is_voice_installed(info) else _("(Niet geïnstalleerd)")
            voice_names.append(f"{info.voice_id} ({info.lang}) - {status}")
        self.sList.Set(voice_names)
        
        self.onVoiceSelect(None) 
        
        # Gebruik ui.message() voor de gesproken feedback
        ui.message(message) 
        
        dlg = wx.MessageDialog(self, message, 
                               _("Operatie voltooid") if success else _("Operatie mislukt"), 
                               wx.OK | (wx.ICON_INFORMATION if success else wx.ICON_ERROR))
        dlg.ShowModal()
        dlg.Destroy()
        
        self.onVoiceSelect(None)
        

    # --- Knoophandlers ---

    def onDownload(self, evt):
        selected_index = self.sList.GetSelection()
        if selected_index == wx.NOT_FOUND: return
        voice_info = self.voices[selected_index]
        
        self._execute_async_task(
            task_func=self._download_voice, 
            success_msg=_("Download is voltooid. De stem is nu beschikbaar in de synthesizer-instellingen."),
            voice_id=voice_info.voice_id,
            refresh_synthesizer=True
        )

    def onDelete(self, evt):
        selected_index = self.sList.GetSelection()
        if selected_index == wx.NOT_FOUND: return
        voice_info = self.voices[selected_index]
        
        self._execute_async_task(
            task_func=self._delete_voice, 
            success_msg=_("De stem is succesvol verwijderd. De synthesizer-stemmenlijst is bijgewerkt."),
            voice_id=voice_info.voice_id,
            refresh_synthesizer=True
        )

    # def onSetDefault(self, evt): # VERWIJDERD (Voormalig 18 regels code verwijderd)

    def onSave(self):
        """Vereiste methode voor SettingsPanel. Acties worden direct uitgevoerd."""
        pass
        
    @script(gesture="kb:NVDA+shift+v") 
    def script_showVoiceManagerPanel(self, gesture):
        """Toont het NVDA-instellingenvenster, met een focus op het Phoonnx-paneel."""
        gui.mainFrame.runSettingsDialog(gui.settingsDialogs.NVDASettingsDialog,
                                        startCategory=PhoonnxVoiceManagerPanel)