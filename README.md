# Qwen2.5-Coder Tool-Use SFT Data Pipeline

A clean, end-to-end data preprocessing and verification pipeline for Supervised Fine-Tuning (SFT) of **Qwen2.5-Coder-3B-Instruct** on multi-turn tool-calling and reasoning datasets. 

This pipeline handles chat template formatting (with tool definitions and thinking tokens enabled), custom loss masking so the model is trained strictly on assistant responses, and automated integrity checks before model training.

---

## 📋 Features

* **Chat & Tool Template Formatting:** Applies the native Qwen ChatML template while injecting JSON tool schemas and enabling thinking tags (`enable_thinking=True`).
* **Assistant-Only Loss Masking:** Automatically locates `<|im_start|>assistant\n` and `<|im_end|>` token boundaries to mask system prompts, tool definitions, and user turns with `-100` (PyTorch's ignore index).
* **Built-in Pipeline Verification:** Includes a 3-step diagnostic suite that decodes tokenized samples to verify masking ratios, tests dynamic batch padding via `DataCollatorForSeq2Seq`, and ensures custom label masks survive collation.

---

## 📁 Directory Structure

Ensure your workspace matches the expected directory layout before running the script:

```text
├── dataset/
│   ├── tools.json                      # JSON file containing tool/function schemas
│   ├── raw/
│   │   ├── train.jsonl                 # Raw conversational data ({"messages": [...]})
│   │   ├── val.jsonl
│   │   └── test.jsonl
│   ├── processed_untokenized/          # Intermediate ChatML-formatted text files
│   │   ├── train.jsonl
│   │   └── val.jsonl
│   └── processed_tokenized/            # Final tokenized files with input_ids, attention_mask, labels
│       ├── train.jsonl
│       └── val.jsonl
└── scripts/
    └── pipeline.py                     # Main preprocessing and verification script
```

---

## 🛠️ Prerequisites & Installation

Install the required Hugging Face libraries:

```bash
pip install transformers datasets torch
```

---

## 📄 Input Data Format

### 1. `raw/*.jsonl`
Each line in your raw `.jsonl` files must be a JSON object containing a `"messages"` list formatted for standard chat templates:
```json
{"messages": [{"role": "user", "content": "Check the weather in Tokyo."}, {"role": "assistant", "content": "..."}]}
```

### 2. `tools.json`
A valid JSON array of tool definitions compatible with Hugging Face's `apply_chat_template(..., tools=tools)`.

---

## 🚀 Usage

1. **Create the output directories** (the script opens target files in append mode `"a"`, so directories must exist and be empty before a fresh run):
   ```bash
   mkdir -p ../dataset/processed_untokenized ../dataset/processed_tokenized
   ```

2. **Run the pipeline:**
   ```bash
   python pipeline.py
   ```

---

## 🔍 How It Works

### 1. Untokenized Formatting (`converto`)
Loads raw conversations via `loader()` and renders them into raw text strings using `tokenizer.apply_chat_template()`. This embeds the tool schemas and ChatML control tokens (`<|im_start|>`, `<|im_end|>`) without converting them to token IDs yet.

### 2. Tokenization & Loss Masking (`converter` & `build_labels`)
* Tokenizes the formatted text up to `max_seq_len = 1024`.
* Scans the `input_ids` sequence for assistant turn boundaries:
  * **Masked (`-100`):** System prompt, tool definitions, user messages, `<|im_start|>assistant\n` prefix, and `<|im_end|>` token.
  * **Unmasked (Token IDs):** Only the tokens generated *inside* the assistant's response.

### 3. Verification Suite
After generating the Hugging Face `Dataset` and initializing `DataCollatorForSeq2Seq`, the script runs three automated checks:
* **Step 1 (Raw Tokenization Check):** Decodes the first training sample's `input_ids` and `labels` back to text and prints the exact percentage of tokens being trained on.
* **Step 2 (Dynamic Padding Check):** Passes a 5-sample batch through `DataCollatorForSeq2Seq` and asserts that `input_ids`, `attention_mask`, and `labels` share identical rectangular tensor shapes.
* **Step 3 (Mask Preservation Check):** Verifies that prompt tokens remain `-100`, assistant response tokens retain their original IDs, and newly added right-padding tokens are properly masked with `-100`.

---

## ⚠️ Important Notes

* **Append Mode:** Both `converto()` and `converter()` open output files using `"a"` (append mode). If you re-run the script, delete or clear existing files in `processed_untokenized/` and `processed_tokenized/` first to prevent duplicate entries.
* **Sequence Length:** The default `max_seq_len` is set to `1024`. If your tool schemas and multi-turn conversations exceed this limit, increase `max_seq_len` in the `converter()` calls to prevent assistant responses from being truncated.
