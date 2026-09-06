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
from tokenizers import Tokenizer

def find_onnx_dir():
    candidates = [
        os.path.join(os.path.dirname(__file__), "onnx_model"),
        os.path.join(os.getcwd(), "onnx_model"),
        os.path.join(os.path.dirname(__file__), "..", "public", "models", "Xenova"),
        os.path.join(os.getcwd(), "public", "models", "Xenova")
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "model_quantized.onnx")):
            return c
    return candidates[0]

ONNX_DIR = find_onnx_dir()
MODEL_PATH = os.path.join(ONNX_DIR, "model_quantized.onnx")
TOKENIZER_PATH = os.path.join(ONNX_DIR, "tokenizer.json")

LABEL_LIST = ["O", "B-NAME", "I-NAME", "B-ADDRESS", "I-ADDRESS"]
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

def test_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[!] Model not found at {MODEL_PATH}.")
        return

    print(f"Loading Quantized Model from {MODEL_PATH}...")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    tokenizer.no_padding()
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
    print("RUNNING INFERENCE BENCHMARKS (Fine-Tuned MiniLM-L6 INT8 ONNX)")
    print("="*70)

    for text in test_sentences:
        encoding = tokenizer.encode(text)
        input_ids = np.array([encoding.ids], dtype=np.int64)
        attention_mask = np.array([encoding.attention_mask], dtype=np.int64)

        t0 = time.perf_counter()
        outputs = session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })
        elapsed_ms = (time.perf_counter() - t0) * 1000

        logits = outputs[0][0]
        preds = np.argmax(logits, axis=-1)
        tokens = encoding.tokens

        entities = []
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
