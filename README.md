# Paired Cardiac Cell with Drug Exposure Parameter Estimation

Code accompanying the paper:
**"Mechanistic Modelling of Anti-Arrhythmic Drug Action from Paired Cardiac Cell Recordings"**
Zhechao Yang, University of Glasgow, 2026.

## Requirements
Install dependencies with:
    pip install -r requirements.txt

## Installation

### Python Requirements
Python 3.6 or newer is required.

Install standard dependencies:
```bash
pip install -r requirements.txt
```

---

### Installing PINTS

The latest release of PINTS can be installed with pip:
```bash
pip install --upgrade pip
pip install pints
```

Alternatively, to install the latest cutting-edge version directly from the repository:
```bash
git clone https://github.com/pints-team/pints.git
cd pints
pip install -e .[dev,docs]
```

To uninstall:
```bash
pip uninstall pints
```

---

### Installing Myokit

Myokit installation requires three steps:

1. **Download and install Miniconda** (Python 3, 64-bit):  
   https://conda.io/miniconda.html

2. **Download and install a C++ compiler.**  
   The easiest option is [Visual Studio 2022](https://visualstudio.microsoft.com/downloads/).  
   When installing, select **"C++ build tools"** and enable the **"MSVC"** and **"Windows 10/11 SDK"** optional features.  
   See https://wiki.python.org/moin/WindowsCompilers for further compatibility information.

3. **Open an Anaconda prompt as administrator** and type:
```bash
pip install myokit[pyqt]
```

To add start menu icons:
```bash
myokit icons
```

To upgrade an existing Myokit installation:
```bash
pip install --upgrade myokit
```

---

### Shannon Model

This code uses the **Shannon–Wang–Puglisi–Weber–Bers (2004)** rabbit ventricular myocyte model.  
The original model is available from the CellML Model Repository:  
http://models.cellml.org/electrophysiology

The modified version used in this work, which runs to recreate the published results, is available here:  
https://models.cellml.org/exposure/d72a36fe0b7e121068c96bcb1ff6044a/shannon_wang_puglisi_weber_bers_2004_a.cellml/view

Download the `.cellml` file and convert it to Myokit's `.mmt` format using:
```bash
myokit convert shannon_wang_puglisi_weber_bers_2004_a.cellml
```
Place the resulting `.mmt` file in the directory expected by the scripts before running.

## Usage
1. Run `updated_a01_fit_models_to_data_auto.py` to fit the model to paired recordings
2. Run `updated_fit_plot_test` to generate model vs experiment figures

## Data
Experimental recordings are not included. Contact the authors for access.

