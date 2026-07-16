"""NVDA settings panel for downloading and removing phoonnx voices.

Registers a "Phoonnx Voices" category in NVDA Settings that lists the phoonnx
voice catalog, downloads voices into the phoonnx cache directory (where the
synth driver discovers them at voice-switch time) and removes installed ones.
All network work runs on background threads; the UI is updated via
``wx.CallAfter``.
"""
import importlib.util
import os
import threading

import globalPluginHandler
import gui
import wx
from gui import guiHelper
from gui.settingsDialogs import NVDASettingsDialog, SettingsPanel
from logHandler import log

try:
    import addonHandler
    addonHandler.initTranslation()
except Exception:
    _ = lambda s: s

_ADDON_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_VOICE_MANAGER_PATH = os.path.join(_ADDON_ROOT, "synthDrivers", "phoonnx", "voice_manager.py")


def _load_voice_manager():
    spec = importlib.util.spec_from_file_location("phoonnx_voice_manager", _VOICE_MANAGER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PhoonnxVoicesPanel(SettingsPanel):
    title = _("Phoonnx Voices")

    def makeSettings(self, settingsSizer):
        self._vm = _load_voice_manager()
        self._catalog = []
        sHelper = guiHelper.BoxSizerHelper(self, sizer=settingsSizer)

        self._status = wx.StaticText(self, label=_("Loading voice catalog..."))
        sHelper.addItem(self._status)

        self._voiceList = sHelper.addLabeledControl(
            _("Available voices:"), wx.ListBox, choices=[])

        buttons = guiHelper.ButtonHelper(wx.HORIZONTAL)
        self._downloadButton = buttons.addButton(self, label=_("&Download"))
        self._downloadButton.Bind(wx.EVT_BUTTON, self.onDownload)
        self._removeButton = buttons.addButton(self, label=_("&Remove"))
        self._removeButton.Bind(wx.EVT_BUTTON, self.onRemove)
        refreshButton = buttons.addButton(self, label=_("Re&fresh catalog"))
        refreshButton.Bind(wx.EVT_BUTTON, lambda evt: self._loadCatalog())
        sHelper.addItem(buttons)

        self._loadCatalog()

    # --- catalog ---

    def _loadCatalog(self):
        self._status.SetLabel(_("Loading voice catalog..."))
        threading.Thread(target=self._fetchCatalog, daemon=True).start()

    def _fetchCatalog(self):
        try:
            catalog = self._vm.list_catalog()
        except Exception as e:
            log.error(f"Phoonnx voice manager: catalog fetch failed: {e}")
            wx.CallAfter(self._status.SetLabel,
                         _("Could not load the voice catalog: {error}").format(error=e))
            return
        wx.CallAfter(self._populate, catalog)

    def _populate(self, catalog):
        installed = set(self._vm.installed_voices())
        self._catalog = catalog
        self._voiceList.Clear()
        for entry in catalog:
            mark = _(" [installed]") if entry["id"] in installed else ""
            self._voiceList.Append(f"{entry['id']} ({entry['lang']}){mark}")
        self._status.SetLabel(
            _("{count} voices available, {installed} installed.").format(
                count=len(catalog), installed=len(installed)))

    def _selectedEntry(self):
        idx = self._voiceList.GetSelection()
        if idx == wx.NOT_FOUND or idx >= len(self._catalog):
            return None
        return self._catalog[idx]

    # --- actions ---

    def onDownload(self, evt):
        entry = self._selectedEntry()
        if entry is None:
            return
        self._downloadButton.Disable()
        self._status.SetLabel(_("Downloading {voice}...").format(voice=entry["id"]))
        threading.Thread(target=self._download, args=(entry,), daemon=True).start()

    def _download(self, entry):
        def progress(done, total):
            if total:
                pct = int(done * 100 / total)
                wx.CallAfter(self._status.SetLabel,
                             _("Downloading {voice}: {pct}%").format(voice=entry["id"], pct=pct))

        try:
            self._vm.download_voice(entry, progress=progress)
        except Exception as e:
            log.error(f"Phoonnx voice manager: download failed: {e}")
            wx.CallAfter(self._status.SetLabel,
                         _("Download of {voice} failed: {error}").format(voice=entry["id"], error=e))
        else:
            wx.CallAfter(self._status.SetLabel,
                         _("{voice} installed. Switch synthesizers to refresh the voice list.").format(
                             voice=entry["id"]))
            wx.CallAfter(self._populate, self._catalog)
        finally:
            wx.CallAfter(self._downloadButton.Enable)

    def onRemove(self, evt):
        entry = self._selectedEntry()
        if entry is None:
            return
        voice_id = entry["id"].replace("/", "_").replace("\\", "_")
        if voice_id not in set(self._vm.installed_voices()):
            self._status.SetLabel(_("{voice} is not installed.").format(voice=entry["id"]))
            return
        try:
            self._vm.remove_voice(voice_id)
        except Exception as e:
            self._status.SetLabel(_("Could not remove {voice}: {error}").format(
                voice=entry["id"], error=e))
            return
        self._status.SetLabel(_("{voice} removed.").format(voice=entry["id"]))
        self._populate(self._catalog)

    def onSave(self):
        pass


class GlobalPlugin(globalPluginHandler.GlobalPlugin):
    def __init__(self):
        super().__init__()
        NVDASettingsDialog.categoryClasses.append(PhoonnxVoicesPanel)

    def terminate(self):
        try:
            NVDASettingsDialog.categoryClasses.remove(PhoonnxVoicesPanel)
        except ValueError:
            pass
        super().terminate()
