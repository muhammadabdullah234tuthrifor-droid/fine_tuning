import json

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)

tools_path = "../dataset/tools.json"
raw_train_path = "../dataset/raw/train.jsonl"
raw_val_path = "../dataset/raw/val.jsonl"
raw_test_path = "../dataset/raw/test.jsonl"

processed_untokenized_train_path = "../dataset/processed_untokenized/train.jsonl"
processed_untokenized_val_path = "../dataset/processed_untokenized/val.jsonl"
processed_untokenized_test_path = "../dataset/processed_untokenized/test.jsonl"

processed_tokenized_train_path = "../dataset/processed_tokenized/train.jsonl"
processed_tokenized_val_path = "../dataset/processed_tokenized/val.jsonl"
processed_tokenized_test_path = "../dataset/processed_tokenized/test.jsonl"

model_id = "Qwen/Qwen2.5-Coder-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="right")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

with open(tools_path, "r") as f:
    tools = json.load(f)

assistant_prefix = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
im_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)


def find_subsequence(sequence, pattern, start=0):
    n = len(pattern)
    for i in range(start, len(sequence) - n + 1):
        if sequence[i : i + n] == pattern:
            return i
    return -1


def build_labels(input_ids, assistant_prefix, im_end):
    labels = [-100] * len(input_ids)
    pos = 0

    while True:
        start = find_subsequence(input_ids, assistant_prefix, pos)
        if start == -1:
            break

        # Skip "<|im_start|>assistant\n"
        response_start = start + len(assistant_prefix)
        end = find_subsequence(input_ids, im_end, response_start)
        if end == -1:
            end = len(input_ids)

        # Train only on assistant response
        for i in range(response_start, end):
            labels[i] = input_ids[i]

        pos = end + len(im_end)

    return labels


def converto(messages, target_path):
    with open(target_path, "a") as f:
        for i in messages:
            formatted = {
                "text": tokenizer.apply_chat_template(
                    i,
                    tokenize=False,
                    enable_thinking=True,
                    tools=tools,
                    add_generation_prompt=False,
                )
            }
            json.dump(formatted, f)
            f.write("\n")


def converter(path: str, target_path: str, assistant_prefix, im_end, max_seq_len):
    with open(path, "r") as f, open(target_path, "a") as d:
        messages = [json.loads(line.strip())["text"] for line in f]
        for i in messages:
            formatted = tokenizer(
                i, padding=False, truncation=True, max_length=max_seq_len
            )
            input_ids = formatted["input_ids"]
            attention_mask = formatted["attention_mask"]
            labels = build_labels(input_ids, assistant_prefix, im_end)

            sample = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
            json.dump(sample, d)
            d.write("\n")


def loader(path: str) -> list:
    with open(path, "r") as f:
        return [json.loads(line)["messages"] for line in f]


train_messages = loader(raw_train_path)
val_messages = loader(raw_val_path)

converto(train_messages, processed_untokenized_train_path)
converto(val_messages, processed_untokenized_val_path)

converter(
    processed_untokenized_train_path,
    processed_tokenized_train_path,
    assistant_prefix,
    im_end,
    1024,
)
converter(
    processed_untokenized_val_path,
    processed_tokenized_val_path,
    assistant_prefix,
    im_end,
    1024,
)

files_path = {
    "train": processed_tokenized_train_path,
    "validation": processed_tokenized_val_path,
}

dataset = load_dataset("json", data_files=files_path)
train_dataset = dataset["train"]
val_dataset = dataset["validation"]

print(f"dataset: {dataset}")
print(type(dataset))
print(dataset["train"][0])

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    label_pad_token_id=-100,
)
print(data_collator)
print(type(data_collator))

# ==========================================
# COMPREHENSIVE DATA PIPELINE VERIFICATION
# ==========================================


def verify_and_decode(sample):
    input_ids = sample["input_ids"]
    attention_mask = sample["attention_mask"]
    labels = sample["labels"]

    real_input_ids = [
        token_id for token_id, mask in zip(input_ids, attention_mask) if mask == 1
    ]
    decoded_input = tokenizer.decode(real_input_ids, skip_special_tokens=False)

    print("\n--- Decoded Input (Real tokens only) ---")
    print(decoded_input[:1000] + "..." if len(decoded_input) > 1000 else decoded_input)
    print(f"Length of real tokens: {len(real_input_ids)}")

    real_labels = [token_id for token_id in labels if token_id != -100]
    decoded_labels = tokenizer.decode(real_labels, skip_special_tokens=False)

    print("\n--- Decoded Labels (Target tokens) ---")
    print(
        decoded_labels[:1000] + "..." if len(decoded_labels) > 1000 else decoded_labels
    )
    print(f"Number of target tokens: {len(real_labels)}")

    total_tokens = len(labels)
    masked_tokens = labels.count(-100)
    unmasked_ratio = (len(real_labels) / total_tokens) * 100
    print(
        f"Masking Efficiency: {unmasked_ratio:.1f}% trained on "
        f"(Total: {total_tokens} | Masked: {masked_tokens} | Learning: {len(real_labels)})"
    )


# STEP 1: Verify Raw Tokenization & Label Masking
with open(processed_tokenized_train_path, "r") as f:
    first_tokenized_line = f.readline()
    if first_tokenized_line:
        verify_and_decode(json.loads(first_tokenized_line))
    else:
        print("Error: The processed file is empty!")


# STEP 2: Verify Dynamic Padding & Label Preservation
fake_batch = [train_dataset[i] for i in range(5)]

print(f"\nOriginal Sample 0 length: {len(fake_batch[0]['input_ids'])} tokens")
print(f"Original Sample 1 length: {len(fake_batch[1]['input_ids'])} tokens")

padded_batch = data_collator(fake_batch)

print(f"Padded Input IDs shape: {tuple(padded_batch['input_ids'].shape)}")
print(f"Padded Attention Mask shape: {tuple(padded_batch['attention_mask'].shape)}")
print(f"Padded Labels shape: {tuple(padded_batch['labels'].shape)}")

assert (
    padded_batch["input_ids"].shape == padded_batch["labels"].shape
), "Shapes don't match!"


# STEP 3: Verify Custom Label Masks Survived Collation
original_labels = fake_batch[0]["labels"]
padded_labels = padded_batch["labels"][0].tolist()

try:
    first_unmasked_idx = next(i for i, x in enumerate(original_labels) if x != -100)
    original_len = len(original_labels)
    padded_len = len(padded_labels)

    print(f"\nFirst unmasked token index: {first_unmasked_idx}")
    print(
        f"Original label: {original_labels[first_unmasked_idx]} | "
        f"Padded label: {padded_labels[first_unmasked_idx]}"
    )
    print(
        f"Index 0 Original: {original_labels[0]} | Index 0 Padded: {padded_labels[0]}"
    )

    if padded_len > original_len:
        padding_region = padded_labels[original_len:]
        print(f"All padding is -100? {all(x == -100 for x in padding_region)}")

    if (
        original_labels[first_unmasked_idx] == padded_labels[first_unmasked_idx]
        and padded_labels[0] == -100
        and (
            padded_len == original_len
            or all(x == -100 for x in padded_labels[original_len:])
        )
    ):
        print("\nSUCCESS: Pipeline verification passed.")
    else:
        print("\nFAILURE: Label masking mismatch in collator.")

except StopIteration:
    print("ERROR: No unmasked tokens found in the sample.")
