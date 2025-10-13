# Phoonnx TTS synthesizer for NVDA screenreader (proof of concept)

The Phoonnx TTS Driver is an NVDA add-on that integrates the Phoonnx engine as a speech synthesizer. This driver is designed to utilize ONNX-based voices (like PiperTTS) within the NVDA screen reader.

## 📦 Installation (test version)

Install the test add-on manually using the add-on file (`.nvda-addon`).

1.  Download the latest `.nvda-addon` file from my [Google Drive](https://drive.google.com/file/d/1l-bloYDdrpFJWfz4hZ2hGpr-fejTckHw/view?usp=sharing) (tested with NVDA 2025.3, minimal version is 2025.1)
2.  Ensure NVDA is running.
3.  Press Enter on the downloaded `.nvda-addon` file in Windows Explorer.
4.  NVDA will ask if you want to install the add-on. Confirm the installation and follow the prompts.
5.  NVDA will ask you to restart the screen reader. Do this to complete the installation.

Note: espeak-NG has to be installed system-wide to use this add-on at the moment!

## ⚙️ Configuration

After installation, you must select the Phoonnx synthesizer in NVDA:

1.  Open the NVDA Menu (**NVDA key + N**).
2.  Go to Preferences** and then **Synthesizer...** (or Nvda + control + S)
3.  Select Phoonnx TTS Driver" from the synthesizer combo box.
4.  Press OK to save the settings.
5.  You can now adjust the voice, rate, volume, and pitch via NVDA's Speech Settings.

> **Note on Rate:** The add-on translates the NVDA rate setting (0-100) to the TTS model's `length_scale`. A default NVDA rate of **50** corresponds to a `length_scale` of **1.0** (normal speed). Lower rates result in a higher `length_scale` (slower speech), and higher rates result in a lower `length_scale` (faster speech).

## 🛠 Developer Requirements (For Building)

To develop or bundle this add-on, you need to set up a specific Python environment that matches NVDA's requirements.

### 1. Python Environment Setup

NVDA currently uses **Python 3.11.9 (32-bit)**. You must use this exact version to ensure library compatibility.

1.  **Install Python 3.11.9 (32-bit)** for Windows.
2.  **Create a Virtual Environment (venv):**
    ```bash
    py -3.11 -m venv phoonnx_venv
    phoonnx_venv\Scripts\activate
    ```
3.  **Install the Phoonnx Package:**
    ```bash
    pip install phoonnx
    ```

### 2. Bundling Libraries (`phoonnx_libs`)

The add-on bundles the phoonnx dependencies in the `phoonnx_libs` folder .

1.  Copy the relevant contents of your virtual environment's `site-packages` directory (usually `phoonnx_venv\Lib\site-packages`) to the add-on's `phoonnx_libs` folder. See list in the folder in this repo.

2. Due to potential conflicts with packages already included in NVDA's environment, you must replace or delete the files in `phoonnx_libs` that conflict with NVDA's core dependencies. See files in the repo.

### 3. Model Files

Ensure the following model and configuration files are present in the root driver directory:

* `dii_nl-NL.onnx` (The ONNX model file)
* `dii_nl-NL.onnx.json` (The configuration file for the model)
(or download other voice models at: https://huggingface.co/OpenVoiceOS/models)

### 4. System Dependency (espeak-ng)

The `phoonnx` package currently relies on system-wide installation of the **espeak-ng** binary for certain functionalities (e.g., text processing/phonemization).

* You must **install espeak-ng system-wide** on your Windows development/testing machine.
