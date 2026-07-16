# Phoonnx TTS synthesizer for NVDA

An [NVDA](https://www.nvaccess.org/) add-on for Windows that integrates the
[phoonnx engine](https://github.com/TigreGotico/phoonnx) as a speech synthesizer,
bringing ONNX-based neural voices (Piper-style and beyond) to the NVDA screen reader.

Demo video: https://www.youtube.com/watch?v=ASYrV8R1zQw

## Credits

- [@timonvanhasselt](https://github.com/timonvanhasselt) (Visio) — original add-on author.
- [@JarbasAI](https://github.com/JarbasAl) — the phoonnx engine.

## 📦 Installation

1. Download the latest `.nvda-addon` file from the
   [releases section](https://github.com/TigreGotico/phoonnx-AddonNVDA/releases)
   (tested with NVDA 2025.3, minimum version 2025.1).
2. With NVDA running, press Enter on the downloaded `.nvda-addon` file in Windows Explorer.
3. Confirm the installation and restart NVDA when prompted.

## ⚙️ Configuration

1. Open the NVDA menu (**NVDA+N**) → **Preferences** → **Synthesizer...** (or **NVDA+Ctrl+S**).
2. Select "Phoonnx TTS Driver" from the synthesizer combo box and press OK.
3. Adjust the voice, rate and volume via NVDA's Speech Settings.

The add-on currently ships a single bundled voice (`dii_nl-NL`, Dutch). Voice
management (downloading and switching voices at runtime) is on the roadmap.

> **Note on rate:** the NVDA rate setting (0–100) maps to the model's
> `length_scale`; rate **50** is normal speed (`length_scale` 1.0), higher rates
> are faster. The mapping is clamped to a usable range at both extremes.

## 🛠 Developing

All driver logic lives in `synthDrivers/phoonnx/__init__.py`. Text is queued by
`SynthDriver.speak()` and synthesized on a worker thread that streams int16 audio
chunks to an `nvwave.WavePlayer`; index and break commands in the speech sequence
are honored, and the NVDA UI thread is never blocked.

### Python environment

NVDA embeds its own Python; the bundled libraries must match it exactly:

- NVDA 2025.x: Python **3.11.9 (32-bit)**
- NVDA 2026.x: Python **3.13.x (64-bit)**

```bash
py -m venv phoonnx_venv
phoonnx_venv\Scripts\activate
pip install phoonnx
```

### Bundling libraries (`phoonnx_libs`)

Copy the relevant contents of the venv's `site-packages` into
`synthDrivers/phoonnx/phoonnx_libs/` (added to `sys.path` at import time; not
committed to the repo). Place the voice model (`dii_nl-NL.onnx` +
`dii_nl-NL.onnx.json`) next to `__init__.py`; only the `.onnx.json` is committed.

Voices whose config uses `"phoneme_type": "espeak"` need espeak phonemization at
runtime. Bundling the espeak-ng binary on Windows is fragile;
[espyak](https://github.com/TigreGotico/espyak), a pure-Python byte-exact
reimplementation of espeak-ng's G2P, is the planned replacement so no native
binary needs to be packaged.

### Building the package

```bash
python build_addon.py [output_dir]
```

This zips the manifest and `synthDrivers/` into `phoonnx-<version>.nvda-addon`.
It warns if `phoonnx_libs/` or the `.onnx` weights are missing — the resulting
skeleton installs but the driver's `check()` fails until they are added.

### Tests

```bash
pip install pytest numpy
pytest test/
```

The suite stubs the NVDA host modules (`nvwave`, `synthDriverHandler`, …) and the
phoonnx runtime, so it runs on any platform without NVDA or ONNX models. CI runs
it on Python 3.11 and 3.13 (the NVDA interpreter versions) and uploads a skeleton
`.nvda-addon` artifact.
