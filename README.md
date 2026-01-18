# Phoonnx TTS synthesizer for NVDA screenreader (proof of concept)

The Phoonnx TTS Driver is an NVDA add-on for Windows NVDA screenreader that integrates the [Phoonnx engine](https://github.com/TigreGotico/phoonnx) as a speech synthesizer. This driver is designed to utilize ONNX-based voices (like PiperTTS) within the NVDA screen reader.

Check the following youtube video to see it in action: https://www.youtube.com/watch?v=ASYrV8R1zQw

## Credits: 
[@JarbasAI](https://github.com/JarbasAl) of for making the phoonnx engine!

## 📦 Installation (test version)

Install the test add-on manually using the add-on file (`.nvda-addon`).

1.  Download the latest `.nvda-addon` file from the [releases section](https://github.com/timonvanhasselt/phoonnx-AddonNVDA/releases) (tested with NVDA 2025.3, minimal version is 2025.1)
2.  Ensure NVDA is running.
3.  Press Enter on the downloaded `.nvda-addon` file in Windows Explorer.
4.  NVDA will ask if you want to install the add-on. Confirm the installation and follow the prompts.
5.  NVDA will ask you to restart the screen reader. Do this to complete the installation.

## ⚙️ Configuration

After installation, you must select the Phoonnx synthesizer in NVDA:

1.  Open the NVDA Menu (**NVDA key + N**).
2.  Go to Preferences** and then **Synthesizer...** (or Nvda + control + S)
3.  Select Phoonnx TTS Driver" from the synthesizer combo box.
4.  Press OK to save the settings.
5.  You can now adjust the voice, rate, volume, and pitch via NVDA's Speech Settings.
6.  Choose the Phoonnx Voice Settings panel to download more/other voices, update the voice list and/or remove voices from the cache. Voice models are stored in the users folder, for example C:\user\.cache\phoonnx\voices

> **Note on Rate:** The add-on translates the NVDA rate setting (0-100) to the TTS model's `length_scale`. A default NVDA rate of **50** corresponds to a `length_scale` of **1.0** (normal speed). Lower rates result in a higher `length_scale` (slower speech), and higher rates result in a lower `length_scale` (faster speech).

## 🛠 Developer Requirements (For Building)

To develop or bundle this add-on, you need to set up a specific Python environment that matches NVDA's requirements.

### 1. Python Environment Setup

NVDA currently uses **Python 3.11.9 (32-bit for NVDA 2025.x)** or **Python 3.13.x (64-bit for NVDA 2026.x). You must use this exact version to ensure library compatibility.

1.  Install Python for Windows.
2.  Create a Virtual Environment (venv):
    ```bash
    py -m venv phoonnx_venv
    phoonnx_venv\Scripts\activate
    ```
3.  **Install the Phoonnx Package:**
    ```bash
    pip install phoonnx 
    ```
    or `pip install git+https://github.com/TigreGotico/phoonnx` for the pre-releases

### 2. Bundling Libraries (`phoonnx_libs`)

The add-on bundles the phoonnx dependencies in the `phoonnx_libs` folder .

Copy the relevant contents of your virtual environment's `site-packages` directory (usually `phoonnx_venv\Lib\site-packages`) to the add-on's `phoonnx_libs` folder.


