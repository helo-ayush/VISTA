"""
Fine-Tuning Script for In-Browser MiniLM-L6 PII Token Classifier
Specialized exclusively for:
  - NAME (B-NAME, I-NAME)
  - ADDRESS (B-ADDR, I-ADDR)
  - O (Non-PII text)

Supports:
  - 100% resilient: Works with native PyTorch Dataset (no external `datasets` lib strictly required)
  - Fast tokenization and subword alignment
  - Automated ONNX export & dynamic INT8 quantization (~25 MB output)
"""

import os
import json
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

LABEL_LIST = ["O", "B-NAME", "I-NAME", "B-ADDR", "I-ADDR"]
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}

BASE_MODEL = "microsoft/MiniLM-L6-v2"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_model")
ONNX_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "onnx_model")

class PiiJsonlDataset(TorchDataset):
    """
    Pure PyTorch Dataset that loads directly from JSONL without needing
    heavy external libraries (pyarrow, datasets, pandas).
    """
    def __init__(self, filepath, tokenizer, max_samples=None):
        self.tokenizer = tokenizer
        self.records = []
        
        print(f"Reading {filepath}...")
        with open(filepath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if max_samples and idx >= max_samples:
                    break
                rec = json.loads(line.strip())
                int_tags = [LABEL2ID.get(t, 0) for t in rec["ner_tags"]]
                self.records.append((rec["tokens"], int_tags))

        print(f"Loaded {len(self.records):,} samples.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        tokens, ner_tags = self.records[idx]

        tokenized = self.tokenizer(
            tokens,
            truncation=True,
            is_split_into_words=True,
            max_length=256
        )

        word_ids = tokenized.word_ids()
        label_ids = []
        prev_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != prev_word_idx:
                label_ids.append(ner_tags[word_idx] if word_idx < len(ner_tags) else -100)
            else:
                # Subword continuation (B-TAG becomes I-TAG)
                orig_tag_id = ner_tags[word_idx] if word_idx < len(ner_tags) else -100
                if orig_tag_id != -100:
                    tag_name = ID2LABEL[orig_tag_id]
                    if tag_name.startswith("B-"):
                        tag_name = "I-" + tag_name[2:]
                    label_ids.append(LABEL2ID[tag_name])
                else:
                    label_ids.append(-100)
            prev_word_idx = word_idx

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": label_ids
        }

def export_to_onnx(pytorch_model_dir, onnx_dir):
    """Exports trained PyTorch checkpoint to ONNX and quantizes to INT8 (~25MB)"""
    print("\n" + "="*60)
    print("Exporting Fine-Tuned Model to ONNX & INT8 Quantization")
    print("="*60)
    os.makedirs(onnx_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(pytorch_model_dir)
    model = AutoModelForTokenClassification.from_pretrained(pytorch_model_dir)
    model.eval()

    dummy_input = tokenizer("Customer: Aarav Sharma, Flat 101, Delhi 110085", return_tensors="pt")

    raw_onnx_path = os.path.join(onnx_dir, "model.onnx")
    quantized_onnx_path = os.path.join(onnx_dir, "model_quantized.onnx")

    torch.onnx.export(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        raw_onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"}
        },
        opset_version=14
    )
    print(f"[OK] Base ONNX model exported to: {raw_onnx_path}")

    # Apply dynamic INT8 quantization for browser WASM speed
    print("Applying dynamic INT8 quantization for browser WASM execution...")
    quantize_dynamic(
        model_input=raw_onnx_path,
        model_output=quantized_onnx_path,
        weight_type=QuantType.QInt8
    )

    size_mb = os.path.getsize(quantized_onnx_path) / (1024 * 1024)
    print(f"[OK] Quantized Model ready: {quantized_onnx_path} ({size_mb:.1f} MB)")

    # Save tokenizer assets into onnx_dir so it can be dropped straight into public/models/
    tokenizer.save_pretrained(onnx_dir)
    print("[OK] Tokenizer configuration saved into onnx folder.")

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    train_file = os.path.join(data_dir, "train.jsonl")
    val_file = os.path.join(data_dir, "val.jsonl")

    if not os.path.exists(train_file):
        print("Data not found. Running dataset generator first...")
        from generate_dataset import main as gen_main
        gen_main()

    print(f"Loading Base Tokenizer & Model: {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    # Load datasets
    train_dataset = PiiJsonlDataset(train_file, tokenizer)
    val_dataset = PiiJsonlDataset(val_file, tokenizer)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nTraining device: {device_str.upper()}")

    # Determine batch size:
    # On CPU, batch size 16 allows smooth execution without RAM pressure
    # On GPU, batch size 32 is optimal
    batch_size = 32 if torch.cuda.is_available() else 16

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("\n" + "="*60)
    print("Starting Fine-Tuning MiniLM-L6 on 100K Indian PII Dataset...")
    print("="*60)
    trainer.train()

    print(f"\nSaving PyTorch model checkpoint to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Export directly to quantized browser ONNX
    export_to_onnx(OUTPUT_DIR, ONNX_OUTPUT_DIR)
    print("\n[SUCCESS] Model training and browser ONNX export complete!")
    print(f"To deploy, copy contents of '{ONNX_OUTPUT_DIR}' to 'public/models/Xenova/'")

if __name__ == "__main__":
    main()
