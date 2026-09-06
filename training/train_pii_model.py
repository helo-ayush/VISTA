"""
Fine-Tuning Script for In-Browser MiniLM-L6 PII Token Classifier
Specialized for:
  - NAME (B-NAME, I-NAME)
  - ADDRESS (B-ADDR, I-ADDR)
  - O (Non-PII text)

Supports:
  - High-speed training on PyTorch (GPU or CPU)
  - Subword label alignment with fast tokenizer
  - Automated ONNX export & INT8 quantization
  - Output size: ~25 MB (Ideal for client-side WASM browser inference)
"""

import os
import json
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# Label definitions
LABEL_LIST = ["O", "B-NAME", "I-NAME", "B-ADDR", "I-ADDR"]
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}

# Base model: MiniLM-L6 is extraordinarily fast, accurate, and lightweight
BASE_MODEL = "microsoft/MiniLM-L6-v2"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_model")
ONNX_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "onnx_model")

def load_jsonl(filepath, limit=None):
    texts = []
    ner_tags_list = []
    tokens_list = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit and idx >= limit:
                break
            record = json.loads(line.strip())
            texts.append(record["text"])
            tokens_list.append(record["tokens"])
            # Map string tags to integer IDs
            int_tags = [LABEL2ID.get(tag, 0) for tag in record["ner_tags"]]
            ner_tags_list.append(int_tags)

    return Dataset.from_dict({
        "tokens": tokens_list,
        "ner_tags": ner_tags_list
    })

def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=256
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx] if word_idx < len(label) else -100)
            else:
                # Subword continuation: if B-TAG, map to I-TAG
                original_tag_id = label[word_idx] if word_idx < len(label) else -100
                if original_tag_id != -100:
                    tag_name = ID2LABEL[original_tag_id]
                    if tag_name.startswith("B-"):
                        tag_name = "I-" + tag_name[2:]
                    label_ids.append(LABEL2ID[tag_name])
                else:
                    label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def export_to_onnx(pytorch_model_dir, onnx_dir):
    """Exports trained PyTorch checkpoint to ONNX and quantizes to INT8 (~25MB)"""
    print("\n--- Exporting Fine-Tuned Model to ONNX & INT8 Quantization ---")
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
    print(f"✓ Base ONNX model exported to: {raw_onnx_path}")

    # Quantize to INT8
    print("Applying dynamic INT8 quantization for fast browser WASM execution...")
    quantize_dynamic(
        model_input=raw_onnx_path,
        model_output=quantized_onnx_path,
        weight_type=QuantType.QInt8
    )

    size_mb = os.path.getsize(quantized_onnx_path) / (1024 * 1024)
    print(f"✓ Quantized Model ready: {quantized_onnx_path} ({size_mb:.1f} MB)")

    # Save tokenizer assets into onnx_dir so it can be dropped straight into public/models/
    tokenizer.save_pretrained(onnx_dir)
    print("✓ Tokenizer configuration saved.")

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    train_file = os.path.join(data_dir, "train.jsonl")
    val_file = os.path.join(data_dir, "val.jsonl")

    if not os.path.exists(train_file):
        print("Data not found. Running dataset generator first...")
        from generate_dataset import main as gen_main
        gen_main()

    print(f"Loading datasets from {data_dir}...")
    # For fast local testing, limit can be set, or full 95,000 for training
    train_dataset = load_jsonl(train_file)
    val_dataset = load_jsonl(val_file)

    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    print(f"Loading Base Tokenizer & Model: {BASE_MODEL}...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    print("Tokenizing and aligning subword labels...")
    train_tokenized = train_dataset.map(lambda ex: tokenize_and_align_labels(ex, tokenizer), batched=True)
    val_tokenized = val_dataset.map(lambda ex: tokenize_and_align_labels(ex, tokenizer), batched=True)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training device: {device_str.upper()}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=3e-5,
        per_device_train_batch_size=32 if torch.cuda.is_available() else 8,
        per_device_eval_batch_size=32 if torch.cuda.is_available() else 8,
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
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("\nStarting Fine-Tuning...")
    trainer.train()

    print(f"\nSaving PyTorch model checkpoint to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Export to ONNX
    export_to_onnx(OUTPUT_DIR, ONNX_OUTPUT_DIR)
    print("\n[SUCCESS] Model training and browser ONNX export complete!")

if __name__ == "__main__":
    main()
