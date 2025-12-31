# globalPlugins/phoonnx/phoonnxSettingsPanel.py

import os
import sys
import wx
import threading
from logHandler import log
import ui 
from scriptHandler import script
import synthDriverHandler 
from gui.settingsDialogs import SettingsPanel 
import gui 
from pathlib import Path 
import json 
import shutil
import languageHandler

_T = lambda s: s 

# --- Paths ---
USER_HOME = os.path.expanduser("~")
PHOONNX_CACHE_DIR = Path(USER_HOME) / ".cache" / "phoonnx"
VOICE_INSTALL_DIR = PHOONNX_CACHE_DIR / "voices"
VOICE_MANAGER_STATE_FILE = PHOONNX_CACHE_DIR / "voices_cache.json"

ADDON_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PHOONNX_LIBS_PATH = os.path.join(ADDON_ROOT_DIR, 'phoonnx_libs')
VOICE_INDEX_DIR = os.path.join(PHOONNX_LIBS_PATH, 'phoonnx', 'voice_index')

if PHOONNX_LIBS_PATH not in sys.path:
    sys.path.insert(0, PHOONNX_LIBS_PATH)

class PhoonnxVoiceManagerPanel(SettingsPanel):
    title = _T("Phoonnx Voice Manager")

    def makeSettings(self, settingsSizer):
        # Initialize variables to prevent AttributeError during loading
        self.current_voices = []
        self.full_voice_list = []
        
        self.sHelper = gui.guiHelper.BoxSizerHelper(self, sizer=settingsSizer)
        
        from phoonnx.model_manager import TTSModelManager
        self.manager = TTSModelManager(cache_path=str(VOICE_MANAGER_STATE_FILE))
        self.manager.models_dir = str(PHOONNX_CACHE_DIR)
        
        # --- Top bar: Language and Search ---
        filterSizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.lang_map = self._scan_languages_nvda()
        sorted_labels = sorted(self.lang_map.keys())
        
        current_lang_code = languageHandler.getLanguage()
        current_label = languageHandler.getLanguageDescription(current_lang_code) or "English"
        
        default_idx = 0
        for i, label in enumerate(sorted_labels):
            if current_label.split(' (')[0] in label:
                default_idx = i
                break

        self.langCombo = wx.ComboBox(self, wx.ID_ANY, choices=sorted_labels, style=wx.CB_READONLY)
        if sorted_labels: self.langCombo.SetSelection(default_idx)
        
        self.searchCtrl = wx.TextCtrl(self, wx.ID_ANY)
        self.searchCtrl.SetHint(_T("Quick search..."))

        filterSizer.Add(wx.StaticText(self, wx.ID_ANY, _T("Language:")), 0, wx.CENTER | wx.RIGHT, 5)
        filterSizer.Add(self.langCombo, 1, wx.EXPAND | wx.RIGHT, 10)
        filterSizer.Add(wx.StaticText(self, wx.ID_ANY, _T("&Find:")), 0, wx.CENTER | wx.RIGHT, 5)
        filterSizer.Add(self.searchCtrl, 1, wx.EXPAND)
        
        settingsSizer.Add(filterSizer, 0, wx.EXPAND | wx.BOTTOM, 10)

        # --- Voice list ---
        self.sHelper.addItem(wx.StaticText(self, wx.ID_ANY, _T("Available voices:")))
        self.sList = wx.ListCtrl(self, wx.ID_ANY, style=wx.LC_REPORT | wx.LC_SINGLE_SEL, size=(-1, 180))
        self.sList.InsertColumn(0, _T("Voice"), width=250)
        self.sList.InsertColumn(1, _T("Status"), width=150)
        self.sHelper.addItem(self.sList, proportion=0, flag=wx.EXPAND)
        
        self.sDetails = wx.StaticText(self, wx.ID_ANY, _T("Select a voice..."), style=wx.ST_NO_AUTORESIZE)
        self.sHelper.addItem(self.sDetails, proportion=0, flag=wx.EXPAND | wx.ALL, border=5)
        
        # --- Button bar ---
        self.buttonSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.sDownloadButton = wx.Button(self, wx.ID_ANY, _T("&Download"))
        self.sDeleteButton = wx.Button(self, wx.ID_ANY, _T("&Remove"))
        self.buttonSizer.Add(self.sDownloadButton, 0, wx.ALL, 5)
        self.buttonSizer.Add(self.sDeleteButton, 0, wx.ALL, 5)
        settingsSizer.Add(self.buttonSizer, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.ALL, 5)

        self.langCombo.Bind(wx.EVT_COMBOBOX, self.onLanguageChange)
        self.searchCtrl.Bind(wx.EVT_TEXT, self.onSearch)
        self.sList.Bind(wx.EVT_LIST_ITEM_SELECTED, self.onVoiceSelect)
        self.sDownloadButton.Bind(wx.EVT_BUTTON, self.onDownload)
        self.sDeleteButton.Bind(wx.EVT_BUTTON, self.onDelete)

        wx.CallAfter(self.onLanguageChange, None)

    def _scan_languages_nvda(self):
        codes_found = set()
        if os.path.isdir(VOICE_INDEX_DIR):
            for f in os.listdir(VOICE_INDEX_DIR):
                if f.endswith(".json"):
                    try:
                        with open(os.path.join(VOICE_INDEX_DIR, f), 'r', encoding='utf-8') as j:
                            data = json.load(j)
                            for entry in data.values():
                                if isinstance(entry, dict) and 'lang' in entry:
                                    codes_found.add(entry['lang'])
                    except: continue
        
        mapping = {}
        for code in codes_found:
            name = languageHandler.getLanguageDescription(code)
            if name and name != code:
                base_name = name.split(' (')[0]
                if base_name not in mapping: mapping[base_name] = []
                mapping[base_name].append(code)
        return mapping

    def onLanguageChange(self, evt):
        if not self: return
        label = self.langCombo.GetValue()
        target_codes = self.lang_map.get(label, [])
        self.sList.DeleteAllItems()
        self.sList.InsertItem(0, _T("Loading..."))
        
        def load_task():
            from phoonnx.model_manager import TTSModelInfo
            filtered = []
            if os.path.isdir(VOICE_INDEX_DIR):
                for f in os.listdir(VOICE_INDEX_DIR):
                    if not f.endswith(".json"): continue
                    try:
                        with open(os.path.join(VOICE_INDEX_DIR, f), 'r', encoding='utf-8') as j:
                            raw = json.load(j)
                            for vid, d in raw.items():
                                if isinstance(d, dict) and d.get('lang') in target_codes:
                                    info = TTSModelInfo(
                                        voice_id=d.get('voice_id', vid),
                                        lang=d.get('lang'),
                                        model_url=d.get('model_url', ''),
                                        config_url=d.get('config_url', '')
                                    )
                                    info.manager = self.manager
                                    filtered.append(info)
                    except: continue
            self.full_voice_list = sorted(filtered, key=lambda x: x.voice_id)
            wx.CallAfter(self.onSearch, None)
        threading.Thread(target=load_task, daemon=True).start()

    def onSearch(self, evt):
        if not self: return
        if not hasattr(self, 'full_voice_list'): return
        query = self.searchCtrl.GetValue().lower()
        self.current_voices = [v for v in self.full_voice_list if query in v.voice_id.lower()]
        
        self.sList.DeleteAllItems()
        if not self.current_voices:
            self.sList.InsertItem(0, _T("No results."))
        else:
            for idx, v in enumerate(self.current_voices):
                p = VOICE_INSTALL_DIR / v.voice_id
                installed = p.is_dir() and (list(p.glob("*.onnx")) or list(p.glob("*.pt")))
                status_text = _T("Installed") if installed else _T("Available")
                self.sList.InsertItem(idx, v.voice_id.split('/')[-1])
                self.sList.SetItem(idx, 1, status_text)
        self.update_button_states()

    def onVoiceSelect(self, evt):
        if not self: return
        if not hasattr(self, 'current_voices') or not self.current_voices:
            return
            
        idx = self.sList.GetFirstSelected()
        if idx == -1 or idx >= len(self.current_voices): return
        info = self.current_voices[idx]
        p = VOICE_INSTALL_DIR / info.voice_id
        installed = p.is_dir() and (list(p.glob("*.onnx")) or list(p.glob("*.pt")))
        lang_name = languageHandler.getLanguageDescription(info.lang) or info.lang
        self.sDetails.SetLabel(f"ID: {info.voice_id}\nLanguage: {lang_name}\nStatus: {'Installed' if installed else 'Not present locally'}")
        self.update_button_states(info)

    def update_button_states(self, info=None):
        if not self: return
        idx = self.sList.GetFirstSelected()
        if idx == -1 or not hasattr(self, 'current_voices') or not self.current_voices:
            self.sDownloadButton.SetLabel(_T("&Download"))
            self.sDownloadButton.Disable()
            self.sDeleteButton.Disable()
            return
        if not info: info = self.current_voices[idx]
        p = VOICE_INSTALL_DIR / info.voice_id
        installed = p.is_dir() and (list(p.glob("*.onnx")) or list(p.glob("*.pt")))
        self.sDownloadButton.SetLabel(_T("&Update") if installed else _T("&Download"))
        self.sDownloadButton.Enable(True)
        self.sDeleteButton.Enable(bool(installed))

    def onDownload(self, evt):
        idx = self.sList.GetFirstSelected()
        if idx == -1: return
        info = self.current_voices[idx]
        voice_name = info.voice_id.split('/')[-1]

        self.progressDialog = gui.IndeterminateProgressDialog(
            gui.mainFrame,
            _T("Phoonnx"),
            _T("Downloading voice '{name}'...").format(name=voice_name)
        )
        
        def task():
            try:
                info.load()
                wx.CallAfter(self._on_download_finished, True, _T("Finished: {name} has been updated.").format(name=voice_name))
            except Exception as e:
                wx.CallAfter(self._on_download_finished, False, f"Error: {e}")
        
        threading.Thread(target=task, daemon=True).start()

    def _on_download_finished(self, success, msg):
        if hasattr(self, 'progressDialog'):
            self.progressDialog.done()
            del self.progressDialog
        
        if not self: return
        if success:
            self._refresh_and_notify(msg)
        else:
            gui.messageBox(msg, "Phoonnx", wx.OK | wx.ICON_ERROR)
            self.update_button_states()
        
        wx.CallLater(1500, self.sList.SetFocus)

    def onDelete(self, evt):
        idx = self.sList.GetFirstSelected()
        if idx == -1: return
        info = self.current_voices[idx]
        voice_name = info.voice_id.split('/')[-1]
        
        msg = _T("Are you sure you want to remove the voice '{voice}'?").format(voice=voice_name)
        if gui.messageBox(msg, _T("Confirm removal"), wx.YES_NO | wx.ICON_QUESTION) != wx.YES:
            return

        self.progressDialog = gui.IndeterminateProgressDialog(
            gui.mainFrame,
            _T("Phoonnx"),
            _T("Removing voice '{name}'...").format(name=voice_name)
        )
        
        def task():
            try:
                p = VOICE_INSTALL_DIR / info.voice_id
                if p.exists():
                    shutil.rmtree(p)
                wx.CallAfter(self._on_delete_finished, True, _T("Removed: {name} is no longer present.").format(name=voice_name))
            except Exception as e:
                log.error(f"Phoonnx: Error removing voice: {e}")
                wx.CallAfter(self._on_delete_finished, False, f"Error: {e}")
        
        threading.Thread(target=task, daemon=True).start()

    def _on_delete_finished(self, success, msg):
        if hasattr(self, 'progressDialog'):
            self.progressDialog.done()
            del self.progressDialog
        
        if not self: return
        if success:
            self._refresh_and_notify(msg)
        else:
            gui.messageBox(msg, "Phoonnx", wx.OK | wx.ICON_ERROR)
            self.update_button_states()
        
        wx.CallLater(1000, self.sList.SetFocus)

    def _refresh_and_notify(self, msg):
        try:
            synthDriverHandler.setSynth("phoonnx")
        except: pass
        self.onLanguageChange(None)
        ui.message(msg)

    def onSave(self): pass
