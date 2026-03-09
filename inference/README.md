# CloudScan Inference

This directory contains the necessary scripts and dependencies to run the trained RGCN model for inference.

## Prerequisites

It is highly recommended to use a Python virtual environment to prevent dependency conflicts. 

### 1. Set up a Virtual Environment
From your terminal (at `c:\noel\CloudScan\`), run:
```bash
python -m venv venv
```

Activate the virtual environment:
- **Windows (Command Prompt):** `venv\Scripts\activate.bat`
- **Windows (PowerShell):** `venv\Scripts\Activate.ps1`
- **macOS/Linux:** `source venv/bin/activate`

### 2. Install Dependencies
Make sure your virtual environment is activated, then install using the provided requirements file:
```bash
pip install -r inference\requirements.txt
```

*(Note: Depending on your hardware and Python version, you may need to install `torch` and `torch_geometric` following the specialized instructions on their official websites, especially if you plan to use GPU acceleration.)*

## Running the Model

We've provided a `run_model.py` script that takes care of model initialization and loading your `.pth` weights.

From the `c:\noel\CloudScan` directory, you can run:
```bash
python inference\run_model.py
```

### Script Arguments

By default, the script will automatically look for the dataset in `c:\dataset_risk` and the model weights in `RGCN_model\rgcn_model.pth`. You can override these defaults by passing arguments to the script:

```bash
python inference\run_model.py --model "path/to/your/model.pth" --dataset "path/to/dataset"
```

### Run in Windows

while using powershell
```bash
python inference\run_model.py --dataset ${PWD}
```

while using powershell
```bash
python inference\run_model.py --dataset %CD%
``` 