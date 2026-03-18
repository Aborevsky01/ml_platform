from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction
from pathlib import Path

MODEL_ID = "sergeyzh/rubert-mini-frida"
OUTPUT_DIR = Path("rubert_mini_frida_onnx")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    ort_model = ORTModelForFeatureExtraction.from_pretrained(
        MODEL_ID,
        export=True
    )

    ort_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved ONNX model and tokenizer to: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()