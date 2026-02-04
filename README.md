Here is the step-by-step guide to initializing your **ght-qcxr** project with `uv` and VSCode on your Mac 
### **Step 0: Download MedGemma Model with Vision - instruction trained 4b parameters 8 bit quantised

Use your web browser to download all the files from: https://huggingface.co/mlx-community/medgemma-1.5-4b-it-8bit/tree/main

to path/to/ght-qcxr/medgemma_vision

This model was converted to MLX format from google/medgemma-1.5-4b-it using mlx-vlm version 0.3.9.

### **Step 1: Initialize the Project & Environment**

Open your terminal and navigate to your new folder:

```bash
cd path/to/ght-qcxr

```

Initialize the `uv` project and create the virtual environment:

```bash
# 1. Initialize a generic Python project (creates pyproject.toml)
uv init

# 2. Create the virtual environment (.venv folder)
uv venv

```

### **Step 2: Install Dependencies**

Based on the imports in your `cxr_test_1.py` (`mlx_vlm`, `pydicom`, `PIL`, `transformers`, `numpy`), run this command to add and lock specific versions:

```bash
# Installs the specific stack for Apple Silicon vision inference
uv add mlx-vlm pydicom pillow transformers numpy

```

### **Step 3: Configure VSCode**

To make VSCode recognize your `uv` environment (and fix any Pylance "missing import" warnings):

1. **Open the project:**
```bash
code .

```


2. **Select the Interpreter:**
* Press `Cmd` + `Shift` + `P`.
* Type: **"Python: Select Interpreter"**.
* Select the entry marked **`('.venv': venv)`** or **`./.venv/bin/python`**.


*If it doesn't appear automatically:*
* Choose "Enter interpreter path..." -> "Find..." -> Navigate to `ght-qcxr/.venv/bin/python`.


3. **(Optional) Hardcode the setting:**
Create a folder named `.vscode` and a file inside it named `settings.json` with this content:
```json
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "python.analysis.typeCheckingMode": "basic"
}

```



### **Step 4: Verify File Structure**

Ensure your directory structure looks like this so the script can find the model and the X-ray:

```text
ght-qcxr/
├── .venv/                   # Created by uv
├── .vscode/                 # VSCode settings
├── medgemma_vision/         # (Folder you moved containing model files)
├── cxr/                     # <--- MAKE SURE YOU CREATE THIS
│   └── cxr1.dcm             # <--- PLACE YOUR TEST DICOM HERE
├── cxr_test_1.py            # (Your script)
├── pyproject.toml           # Created by uv
└── uv.lock                  # Created by uv

```

### **Step 5: Run the Project**

You can now run your script using `uv run`. This automatically ensures it uses the correct environment variables and dependencies.

```bash
uv run cxr_test_1.py

```

### **Summary of Imports**

* **`mlx-vlm`**: Provides the `load` and `generate` functions optimized for Apple Metal.
* **`pydicom`**: Handles the `.dcm` file reading.
* **`pillow`**: Provides the `Image` module (required for `Image.fromarray`).
* **`transformers`**: Provides the tokenizer utilities.
* **`numpy`**: Handles the pixel array math.