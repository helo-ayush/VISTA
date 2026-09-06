"""
Validation Benchmark Script for MiniLM-L6 INT8 ONNX Token Classifier
Evaluates on all 5,000 samples in training/data/val.jsonl.
Computes:
  - Token-level and Word-level Precision, Recall, F1 for NAME and ADDRESS
  - False positive rate on non-PII negative samples
  - Total latency, average inference time per sample, and throughput
"""

import os
import sys
import json
import time
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

sys.stdout.reconfigure(encoding='utf-8')

VAL_FILE = os.path.join(os.path.dirname(__file__), "data", "val.jsonl")
ONNX_PATH = os.path.join(os.path.dirname(__file__), "..", "public", "models", "vista_pii", "model_quantized.onnx")
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "public", "models", "vista_pii", "tokenizer.json")

LABEL_MAP = {
    0: "O",
    1: "B-NAME",
    2: "I-NAME",
    3: "B-ADDRESS",
    4: "I-ADDRESS"
}

def main():
    if not os.path.exists(VAL_FILE):
        print(f"Error: {VAL_FILE} does not exist!")
        return

    print("=" * 70)
    print(" EVALUATING MODEL ON 5,000 SAMPLES (val.jsonl)")
    print(f" Model: {ONNX_PATH}")
    print("=" * 70)

    t_init = time.perf_counter()
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    tokenizer.no_padding()
    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    print(f"Session loaded in {(time.perf_counter() - t_init):.2f}s")

    # Metrics accumulators
    name_tp = 0
    name_fp = 0
    name_fn = 0

    addr_tp = 0
    addr_fp = 0
    addr_fn = 0

    total_words = 0
    correct_words = 0

    neg_samples = 0
    neg_clean_samples = 0

    total_inference_time = 0.0

    print("Running evaluation on 5,000 validation samples...")

    with open(VAL_FILE, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            rec = json.loads(line)
            tokens = rec["tokens"]
            gold_tags = rec["ner_tags"]

            # Form text
            text = " ".join(tokens)
            is_negative_doc = all(t == "O" for t in gold_tags)
            if is_negative_doc:
                neg_samples += 1

            # Tokenize and run ONNX inference
            encoding = tokenizer.encode(text)
            input_ids = np.array([encoding.ids], dtype=np.int64)
            attention_mask = np.array([encoding.attention_mask], dtype=np.int64)

            t0 = time.perf_counter()
            outputs = session.run(None, {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            })
            total_inference_time += (time.perf_counter() - t0)

            logits = outputs[0][0]
            preds = np.argmax(logits, axis=-1)

            # Map subword predictions back to character offsets
            subword_preds = [LABEL_MAP.get(p, "O") for p in preds]
            offsets = encoding.offsets

            # For each word in gold tokens, find if any subword overlapping with it predicted NAME or ADDRESS
            # Find char ranges of each token in text
            has_doc_fp = False
            char_pos = 0

            for word, gold_tag in zip(tokens, gold_tags):
                total_words += 1
                start_c = text.find(word, char_pos)
                if start_c == -1:
                    start_c = char_pos
                end_c = start_c + len(word)
                char_pos = end_c

                # Check subword predictions covering this range
                pred_category = "O"
                for (s, e), p_tag in zip(offsets, subword_preds):
                    if s == e:
                        continue
                    if max(start_c, s) < min(end_c, e):
                        if "NAME" in p_tag:
                            pred_category = "NAME"
                            break
                        elif "ADDR" in p_tag:
                            pred_category = "ADDRESS"

                gold_category = "NAME" if "NAME" in gold_tag else ("ADDRESS" if "ADDR" in gold_tag else "O")

                if pred_category == gold_category:
                    correct_words += 1

                # Update NAME counts
                if gold_category == "NAME":
                    if pred_category == "NAME":
                        name_tp += 1
                    else:
                        name_fn += 1
                else:
                    if pred_category == "NAME":
                        name_fp += 1
                        has_doc_fp = True

                # Update ADDRESS counts
                if gold_category == "ADDRESS":
                    if pred_category == "ADDRESS":
                        addr_tp += 1
                    else:
                        addr_fn += 1
                else:
                    if pred_category == "ADDRESS":
                        addr_fp += 1
                        has_doc_fp = True

            if is_negative_doc and not has_doc_fp:
                neg_clean_samples += 1

            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1:,} / 5,000 samples ({(idx+1)/50:.0f}%)...")

    # Metrics calculation
    name_p = name_tp / (name_tp + name_fp) if (name_tp + name_fp) > 0 else 0.0
    name_r = name_tp / (name_tp + name_fn) if (name_tp + name_fn) > 0 else 0.0
    name_f1 = (2 * name_p * name_r / (name_p + name_r)) if (name_p + name_r) > 0 else 0.0

    addr_p = addr_tp / (addr_tp + addr_fp) if (addr_tp + addr_fp) > 0 else 0.0
    addr_r = addr_tp / (addr_tp + addr_fn) if (addr_tp + addr_fn) > 0 else 0.0
    addr_f1 = (2 * addr_p * addr_r / (addr_p + addr_r)) if (addr_p + addr_r) > 0 else 0.0

    overall_acc = (correct_words / total_words) * 100 if total_words > 0 else 0.0
    avg_ms = (total_inference_time / 5000) * 1000
    fps = 5000 / total_inference_time if total_inference_time > 0 else 0.0

    print("\n" + "=" * 70)
    print(" VALIDATION RESULTS ON 5,000 SAMPLES")
    print("=" * 70)
    print(f"Total Words Evaluated:  {total_words:,}")
    print(f"Overall Word Accuracy:  {overall_acc:.2f}%\n")

    print(f"{'Category':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'TP':<7} | {'FP':<7} | {'FN':<7}")
    print("-" * 75)
    print(f"{'NAME':<12} | {name_p*100:>8.2f}% | {name_r*100:>8.2f}% | {name_f1*100:>8.2f}% | {name_tp:<7} | {name_fp:<7} | {name_fn:<7}")
    print(f"{'ADDRESS':<12} | {addr_p*100:>8.2f}% | {addr_r*100:>8.2f}% | {addr_f1*100:>8.2f}% | {addr_tp:<7} | {addr_fp:<7} | {addr_fn:<7}")

    print("\n" + "-" * 75)
    print(f"Non-PII Negative Documents: {neg_samples}")
    print(f"Clean Negatives (0 False Positives): {neg_clean_samples} ({(neg_clean_samples/neg_samples*100):.1f}%)")
    print(f"Total Pure Inference Time: {total_inference_time:.2f} seconds")
    print(f"Average Speed per Sample:  {avg_ms:.2f} ms")
    print(f"Throughput:                {fps:.1f} samples/second")
    print("=" * 70)

if __name__ == "__main__":
    main()
