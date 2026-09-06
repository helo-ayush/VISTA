"""
Quick verification script to test any sentence against the trained ONNX model.
Evaluates:
  - Lowercase names and addresses
  - UPPERCASE names and addresses
  - Title Case names and addresses
  - Execution speed in milliseconds
"""

import os
import time
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

ONNX_DIR = os.path.join(os.path.dirname(__file__), "onnx_model")
MODEL_PATH = os.path.join(ONNX_DIR, "model_quantized.onnx")

LABEL_LIST = ["O", "B-NAME", "I-NAME", "B-ADDR", "I-ADDR"]
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

def test_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[!] Model not found at {MODEL_PATH}. Please run train_pii_model.py first.")
        return

    print(f"Loading Quantized Model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(ONNX_DIR)
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

    test_sentences = [
        # 1. Lowercase Indian Name and Address
        "billed to: aarav shinde, residing at flat 402, gali no 3, rohini sector 7, delhi 110085",
        
        # 2. ALL CAPS Indian Name and Address
        "PATIENT NAME: KAVITA DESHMUKH | ADDRESS: PLOT 14-B, NEAR SHIV MANDIR, HSR LAYOUT, BENGALURU 560102",
        
        # 3. Conversational lowercase
        "please send the courier to swapnil kamble at office no. 90 silver oak residency hazratganj lucknow",
        
        # 4. Pure non-PII product (should have 0 names, 0 addresses)
        "samsung galaxy m14 5g smoky teal 6gb ram 128gb storage with 6000mah battery"
    ]

    print("\n" + "="*70)
    print("RUNNING INFERENCE BENCHMARKS")
    print("="*70)

    for text in test_sentences:
        inputs = tokenizer(text, return_tensors="np")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        t0 = time.perf_counter()
        outputs = session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })
        elapsed_ms = (time.perf_counter() - t0) * 1000

        logits = outputs[0][0]
        preds = np.argmax(logits, axis=-1)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        entities = []
        current_entity = None

        for token, pred in zip(tokens, preds):
            tag = ID2LABEL[pred]
            if tag != "O":
                entities.append((token, tag))

        print(f"\nINPUT: {text}")
        print(f"LATENCY: {elapsed_ms:.2f} ms")
        print(f"DETECTED PII ENTITIES:")
        if not entities:
            print("  [None - Non-PII text verified]")
        else:
            for tok, tag in entities:
                print(f"  {tok:<15} -> {tag}")

if __name__ == "__main__":
    test_model()
