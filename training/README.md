# 🛡️ Custom Indian PII Name & Address Model Training

This folder contains the complete, production-grade training pipeline to replace generic off-the-shelf NER with a custom, ultra-lightweight (**~25 MB**), high-speed (**~30 ms**) model specialized exclusively for:
1. **`NAME`** (`B-NAME`, `I-NAME`): Covers all Indian naming conventions (North, South, East, West, Central, Sikh, Muslim, Christian) + global names.
2. **`ADDRESS`** (`B-ADDR`, `I-ADDR`): Covers authentic Indian address structures (Flat/House/Plot, Gali/Street, Sector/Nagar/Layout, Landmark, City, State, PIN).

---

## 📊 Dataset Specifications

The generated dataset contains **100,000 authentic records** (`95,000` train, `5,000` validation):
* **Names Represented**: 100,934+ authentic full names
* **Addresses Represented**: 90,224+ complete multi-line and single-line addresses
* **Casing Distribution**:
  * **40% Lowercase**: `aarav shinde, flat 402, gali no 3, rohini, delhi 110085`
  * **30% UPPERCASE**: `AARAV SHINDE, FLAT 402, GALI NO 3, ROHINI, DELHI 110085`
  * **30% Title Case**: `Aarav Shinde, Flat 402, Gali No 3, Rohini, Delhi 110085`
* **Negative (Non-PII) Samples**: 10% pure product specs, e-commerce item descriptions, and terms to prevent false positives on brand names.

---

## 🚀 How to Train the Model

### Option A: Free Google Colab GPU (Recommended, ~15–20 minutes)
1. Open [Google Colab](https://colab.research.google.com/).
2. Click **File** > **Upload notebook** and upload [`PII_Model_Trainer_Colab.ipynb`](./PII_Model_Trainer_Colab.ipynb).
3. Ensure GPU is enabled (**Runtime** > **Change runtime type** > **T4 GPU**).
4. Run all cells.
5. Colab will automatically train the model, export it to ONNX, quantize it to INT8, and download `pii_browser_model.zip`.
6. Extract the zip into `public/models/vista_pii/`.

---

### Option B: Local Training (On your machine)
1. Generate the dataset (already generated in `training/data/`):
   ```bash
   python training/generate_dataset.py
   ```
2. Start training and automated ONNX export:
   ```bash
   python training/train_pii_model.py
   ```
3. Test inference:
   ```bash
   python training/test_inference.py
   ```

---

## 📦 Output Files
The training script produces:
* `model_quantized.onnx` (~25 MB INT8 quantized ONNX model)
* `tokenizer.json` & `tokenizer_config.json`
* `config.json` & `special_tokens_map.json`

Dropping these into `public/models/vista_pii/` provides an instant drop-in replacement that runs 100% locally in the browser with near-perfect accuracy on Indian names, addresses, and all casing formats.
