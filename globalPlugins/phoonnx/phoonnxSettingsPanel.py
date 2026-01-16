# globalPlugins/phoonnx/phoonnxSettingsPanel.py

import os
import sys
import wx
import threading
import json
from logHandler import log
from pathlib import Path
import shutil
import languageHandler
import gui
import ui 
import synthDriverHandler
from gui.settingsDialogs import SettingsPanel

# --- Paths & Lib setup ---
ADDON_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PHOONNX_LIBS_PATH = os.path.join(ADDON_ROOT_DIR, 'phoonnx_libs')

if PHOONNX_LIBS_PATH not in sys.path:
    sys.path.insert(0, PHOONNX_LIBS_PATH)

from phoonnx.model_manager import TTSModelManager, TTSModelInfo

_T = lambda s: s 

USER_HOME = os.path.expanduser("~")
PHOONNX_CACHE_DIR = Path(USER_HOME) / ".cache" / "phoonnx"
VOICES_JSON_CACHE = PHOONNX_CACHE_DIR / "voices.json"

class PhoonnxVoiceManagerPanel(SettingsPanel):
    title = _T("Phoonnx Voice Manager")

    def makeSettings(self, settingsSizer):
        if not PHOONNX_CACHE_DIR.exists():
            PHOONNX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Language selection
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(wx.StaticText(self, label=_T("Language:")), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        
        self.langCombo = wx.ComboBox(self, choices=[_T("Loading...")], style=wx.CB_READONLY)
        self.langCombo.SetSelection(0)
        self.langCombo.Bind(wx.EVT_COMBOBOX, self.onLanguageChange)
        sizer.Add(self.langCombo, 1, wx.EXPAND)
        settingsSizer.Add(sizer, 0, wx.EXPAND | wx.BOTTOM, 10)

        # Filter field
        filterSizer = wx.BoxSizer(wx.HORIZONTAL)
        filterSizer.Add(wx.StaticText(self, label=_T("&Filter:")), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.filterCtrl = wx.TextCtrl(self)
        self.filterCtrl.Bind(wx.EVT_TEXT, self.onFilterChange)
        filterSizer.Add(self.filterCtrl, 1, wx.EXPAND)
        settingsSizer.Add(filterSizer, 0, wx.EXPAND | wx.BOTTOM, 10)

        # Voice list
        self.sList = wx.ListCtrl(self, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
        self.sList.InsertColumn(0, _T("Status"), width=80)
        self.sList.InsertColumn(1, _T("Voice Name"), width=250)
        self.sList.InsertColumn(2, _T("Engine"), width=120)
        self.sList.Bind(wx.EVT_LIST_ITEM_SELECTED, self.update_button_states)
        settingsSizer.Add(self.sList, 1, wx.EXPAND | wx.BOTTOM, 10)

        btnSizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Update List button
        self.btnUpdateList = wx.Button(self, label=_T("&Update Voice List"))
        self.btnUpdateList.Bind(wx.EVT_BUTTON, self.onUpdateVoiceList)
        btnSizer.Add(self.btnUpdateList, 0, wx.RIGHT, 5)

        # Download button
        self.btnDownload = wx.Button(self, label=_T("&Download"))
        self.btnDownload.Bind(wx.EVT_BUTTON, self.onDownload)
        self.btnDownload.Enable(False)
        btnSizer.Add(self.btnDownload, 0, wx.RIGHT, 5)

        # Remove button
        self.btnDelete = wx.Button(self, label=_T("&Remove"))
        self.btnDelete.Bind(wx.EVT_BUTTON, self.onDelete)
        self.btnDelete.Enable(False)
        btnSizer.Add(self.btnDelete, 0, wx.RIGHT, 5)
        settingsSizer.Add(btnSizer, 0, wx.ALIGN_RIGHT)

        self.current_voices = []
        self.lang_map = {}
        self.manager = None

        threading.Thread(target=self._async_load_data, daemon=True).start()

    def _async_load_data(self):
        try:
            import requests 
            manager = TTSModelManager()
            
            if VOICES_JSON_CACHE.exists():
                try:
                    with open(VOICES_JSON_CACHE, 'r', encoding='utf-8') as f:
                        local_data = json.load(f)
                        for vid, vdata in local_data.items():
                            try:
                                vid_low = str(vid).lower()
                                engine_str = str(vdata.get('engine', '')).lower()
                                # Filter op engine én op de aanwezigheid van 'neurlang' in de ID/naam
                                if 'transformers' in engine_str or 'mimic3' in engine_str or 'neurlang' in vid_low:
                                    continue
                                manager.voices[vid] = TTSModelInfo(**vdata)
                            except Exception:
                                continue
                    log.info(f"Phoonnx Panel: {len(manager.voices)} voices loaded from local cache.")
                except Exception as e:
                    log.error(f"Phoonnx Panel: Error loading local JSON: {e}")
            
            if not manager.voices:
                try:
                    manager.load()
                except Exception as e:
                    log.warning(f"Phoonnx Panel: Initial load skipped/failed: {e}")
                
                self._filter_engines(manager)

            lang_map = self._build_lang_map(manager)
            wx.CallAfter(self._finalize_ui, manager, lang_map)
        except Exception as e:
            log.error(f"Phoonnx Async Load Error: {e}")

    def _filter_engines(self, manager):
        """Removes voice models that are not supported or unwanted."""
        to_remove = []
        for vid, info in manager.voices.items():
            vid_low = str(vid).lower()
            engine_str = str(getattr(info, 'engine', '')).lower()
            
            # Controleer op engine types én op 'neurlang' in de naam
            if 'transformers' in engine_str or 'mimic3' in engine_str or 'neurlang' in vid_low:
                to_remove.append(vid)
                
        for vid in to_remove:
            del manager.voices[vid]

    def _build_lang_map(self, manager):
        """Creates a mapping between readable language names and technical language codes."""
        lang_map = {}
        # Manual translations for codes that NVDA doesn't recognize or handles incorrectly
        custom_translations = {
            "ast": "Asturian",
            "sw-CD": "Swahili (Congo)",
            "tdt-TL": "Tetum",
            "ko-KO": "Korean",
            "sr-RS": "Serbian",
            "vi-VN": "Vietnamese"
        }

        for voice_id, info in manager.voices.items():
            lang = getattr(info, 'lang', 'unknown')
            
            # 1. Check manual translation list
            name = custom_translations.get(lang)
            
            # 2. Check NVDA with the original code (e.g., nl_NL)
            if not name:
                name = languageHandler.getLanguageDescription(lang)
            
            # 3. Fallback: Try swapping underscore/dash
            if not name:
                alt_lang = lang.replace("-", "_") if "-" in lang else lang.replace("_", "-")
                name = languageHandler.getLanguageDescription(alt_lang)
            
            # 4. Ultimate Fallback: Use only the first 2 letters (e.g., 'no' from 'no_NO')
            if not name and len(lang) > 2:
                short_lang = lang[:2]
                name = languageHandler.getLanguageDescription(short_lang)
            
            # 5. If everything fails: show the raw code
            if not name:
                name = lang
                
            if name not in lang_map: lang_map[name] = []
            if lang not in lang_map[name]: lang_map[name].append(lang)
        return lang_map

    def _finalize_ui(self, manager, lang_map):
        self.manager = manager
        self.lang_map = lang_map
        
        sorted_langs = sorted(self.lang_map.keys())
        self.langCombo.SetItems(sorted_langs)
        
        # Try to auto-select current NVDA language
        nvda_lang = languageHandler.getLanguage()
        found_lang = False
        for label, codes in self.lang_map.items():
            if nvda_lang in codes or nvda_lang.split('_')[0] in [c.split('-')[0] for c in codes]:
                self.langCombo.SetValue(label)
                found_lang = True
                break
        
        if not found_lang and sorted_langs:
            self.langCombo.SetSelection(0)
            
        self.onLanguageChange(None)

    def onUpdateVoiceList(self, evt):
        if not self.manager: return
        self.progressDialog = gui.IndeterminateProgressDialog(gui.mainFrame, _T("Phoonnx"), _T("Updating voice list..."))
        
        old_voice_ids = set(self.manager.voices.keys())
        
        def task():
            try:
                try:
                    self.manager.merge_default_voices(store=True)
                except Exception as e:
                    log.warning(f"Phoonnx Panel: Online update encountered errors: {e}")
                
                self._filter_engines(self.manager)
                
                new_voice_ids = set(self.manager.voices.keys())
                added = len(new_voice_ids - old_voice_ids)
                removed = len(old_voice_ids - new_voice_ids)
                
                # Status message
                status_msg = _T("Voice list update completed.           \n\nNew voices: {added}\nRemoved voices: {removed}\nTotal voices: {total}").format(
                    added=added, removed=removed, total=len(new_voice_ids))
                
                new_lang_map = self._build_lang_map(self.manager)
                wx.CallAfter(self._on_update_finished, True, status_msg, new_lang_map)
            except Exception as e:
                log.error(f"Voice list update failure: {e}")
                wx.CallAfter(self._on_update_finished, False, str(e), None)
        
        threading.Thread(target=task, daemon=True).start()

    def _on_update_finished(self, success, msg, new_lang_map):
        if hasattr(self, 'progressDialog'): 
            self.progressDialog.done()
            del self.progressDialog
        
        if success:
            self.lang_map = new_lang_map
            sorted_langs = sorted(self.lang_map.keys())
            current_selection = self.langCombo.GetValue()
            self.langCombo.SetItems(sorted_langs)
            if current_selection in sorted_langs:
                self.langCombo.SetValue(current_selection)
            
            gui.messageBox(msg, _T("Phoonnx Update Result        "), wx.OK | wx.ICON_INFORMATION, parent=self)
            self.onLanguageChange(None)
        else:
            gui.messageBox(msg, _T("Phoonnx Error"), wx.OK | wx.ICON_ERROR, parent=self)

    def onFilterChange(self, evt):
        self.onLanguageChange(None)

    def onLanguageChange(self, evt):
        if not self.manager: return
        
        selected_index = self.sList.GetFirstSelected()
        selected_voice_id = None
        if selected_index != -1 and selected_index < len(self.current_voices):
            selected_voice_id = self.current_voices[selected_index].voice_id

        lang_label = self.langCombo.GetValue()
        target_codes = self.lang_map.get(lang_label, [])
        filter_query = self.filterCtrl.GetValue().lower()
        
        self.sList.Freeze()
        self.sList.DeleteAllItems()
        self.current_voices = []
        
        filtered = [info for info in self.manager.voices.values() if info.lang in target_codes]
        
        if filter_query:
            filtered = [info for info in filtered if filter_query in info.voice_id.lower()]
            
        filtered.sort(key=lambda x: x.voice_id)

        new_selection_index = -1
        for info in filtered:
            idx = self.sList.InsertItem(self.sList.GetItemCount(), "")
            v_id_fixed = info.voice_id.replace("/", os.sep)
            model_path = PHOONNX_CACHE_DIR / "voices" / v_id_fixed / "model.onnx"
            
            installed = model_path.exists()
            status = _T("Yes") if installed else _T("No")
            engine_name = str(getattr(info, 'engine', 'unknown')).split('.')[-1]
            
            self.sList.SetItem(idx, 0, status)
            self.sList.SetItem(idx, 1, info.voice_id)
            self.sList.SetItem(idx, 2, engine_name)
            self.current_voices.append(info)
            
            if info.voice_id == selected_voice_id:
                new_selection_index = idx
        
        self.sList.Thaw()

        if new_selection_index != -1:
            self.sList.Select(new_selection_index)
            self.sList.Focus(new_selection_index)
        
        self.update_button_states()

    def update_button_states(self, evt=None):
        sel = self.sList.GetFirstSelected()
        if sel == -1 or not self.manager:
            self.btnDownload.Enable(False)
            self.btnDelete.Enable(False)
            return
        info = self.current_voices[sel]
        v_id_fixed = info.voice_id.replace("/", os.sep)
        model_path = PHOONNX_CACHE_DIR / "voices" / v_id_fixed / "model.onnx"
        installed = model_path.exists()
        
        self.btnDownload.Enable(not installed)
        self.btnDelete.Enable(installed)

    def onDownload(self, evt):
        sel = self.sList.GetFirstSelected()
        if sel == -1: return
        info = self.current_voices[sel]
        self.progressDialog = gui.IndeterminateProgressDialog(gui.mainFrame, _T("Phoonnx"), _T("Downloading..."))
        
        def task():
            try:
                info.download_config()
                info.download_model()
                if getattr(info, 'tokens_url', None) or getattr(info, 'phoneme_map_url', None):
                    info.download_phoneme_map()
                wx.CallAfter(self._on_download_finished, True, _T("Done"), info.voice_id)
            except Exception as e:
                log.error(f"Download failed: {e}")
                wx.CallAfter(self._on_download_finished, False, str(e), None)
        threading.Thread(target=task, daemon=True).start()

    def _on_download_finished(self, success, msg, voice_id):
        if hasattr(self, 'progressDialog'): 
            self.progressDialog.done()
            del self.progressDialog
        
        if success: 
            self._refresh_and_notify(_T("Download completed"))
            if voice_id:
                self._restart_and_switch(voice_id)
        else: 
            gui.messageBox(msg, "Phoonnx", wx.OK | wx.ICON_ERROR, parent=self)
            wx.CallLater(500, self.sList.SetFocus)

    def onDelete(self, evt):
        sel = self.sList.GetFirstSelected()
        if sel == -1: return
        info = self.current_voices[sel]
        if gui.messageBox(_T("Are you sure you want to remove the model file for this voice?"), _T("Confirm"), wx.YES_NO, parent=self) == wx.YES:
            try:
                v_id_fixed = info.voice_id.replace("/", os.sep)
                voice_dir = PHOONNX_CACHE_DIR / "voices" / v_id_fixed
                model_file = voice_dir / "model.onnx"
                if model_file.exists():
                    os.remove(model_file)
                
                self._refresh_and_notify(_T("Model removed"))
                self._restart_and_switch(None)
            except Exception as e: 
                gui.messageBox(str(e), _T("Error"), parent=self)
                wx.CallLater(500, self.sList.SetFocus)

    def _restart_and_switch(self, voice_id=None):
        try:
            synthDriverHandler.setSynth("phoonnx")
            cur = synthDriverHandler.getSynth()
            if cur and cur.name == "phoonnx" and voice_id:
                cur.voice = voice_id
                ui.message(_T("Voice downloaded and activated: {name}").format(name=voice_id))
            wx.CallLater(3000, self.sList.SetFocus)
        except Exception as e:
            log.error(f"Phoonnx Restart/Switch Error: {e}")
            wx.CallLater(3000, self.sList.SetFocus)

    def _refresh_and_notify(self, msg):
        self.onLanguageChange(None)
        if msg:
            ui.message(msg)

    def onSave(self): pass
