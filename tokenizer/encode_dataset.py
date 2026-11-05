from pathlib import Path
import json

import numpy as np
from tokenizers import Tokenizer


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "tokenizer.json"

DATASET_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "tokenized"

TRAIN_FILE = DATASET_DIR / "train.txt"
VALIDATION_FILE = DATASET_DIR / "validation.txt"

TRAIN_OUTPUT = OUTPUT_DIR / "train.bin"
VALIDATION_OUTPUT = OUTPUT_DIR / "validation.bin"

METADATA_OUTPUT = OUTPUT_DIR / "metadata.json"


# --------------------------------------------------
# Parameters
# --------------------------------------------------

DOCUMENT_END = "<|document_end|>"

DTYPE = np.uint16


# --------------------------------------------------
# Singular file encoding
# --------------------------------------------------

def encode_file(
    tokenizer: Tokenizer,
    input_path: Path,
    output_path: Path,
) -> dict:

    total_documents = 0
    total_tokens = 0
    total_characters = 0

    # Opening binary file
    with input_path.open("r", encoding="utf-8") as source:
        with output_path.open("wb") as output:

            document_lines = []

            def encode_document(lines):
                nonlocal total_documents
                nonlocal total_tokens
                nonlocal total_characters

                if not lines:
                    return

                text = "".join(lines)

                # Text to ID sequence conversion.
                encoding = tokenizer.encode(
                    text,
                    add_special_tokens=False,
                )

                ids = encoding.ids

                if not ids:
                    return

                # Checking ID.
                if max(ids) > np.iinfo(DTYPE).max:
                    raise ValueError(
                        "Token ID does not fit into uint16"
                    )

                # Converting Python list to a more efficient format
                token_array = np.asarray(ids, dtype=DTYPE)

                # Writing the numbers directly to a binary file.
                token_array.tofile(output)

                total_documents += 1
                total_tokens += len(ids)
                total_characters += len(text)

            for line in source:
                document_lines.append(line)

                # Once the end of the document is reached, encode it and clear the temporary buffer.
                if line.strip() == DOCUMENT_END:
                    encode_document(document_lines)
                    document_lines.clear()

            # Process the remainder if the file does not end with an end-of-document marker.
            if document_lines:
                encode_document(document_lines)

    return {
        "documents": total_documents,
        "tokens": total_tokens,
        "characters": total_characters,
        "file_size_bytes": output_path.stat().st_size,
    }


# --------------------------------------------------
# Main function
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer...")

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    vocab_size = tokenizer.get_vocab_size()

    if vocab_size > np.iinfo(DTYPE).max + 1:
        raise ValueError(
            f"Vocabulary size {vocab_size} is too large for uint16"
        )

    print(f"Vocabulary size: {vocab_size}")
    print(f"Storage type: {DTYPE}")

    print("\nEncoding training dataset...")

    train_stats = encode_file(
        tokenizer,
        TRAIN_FILE,
        TRAIN_OUTPUT,
    )

    print("\nEncoding validation dataset...")

    validation_stats = encode_file(
        tokenizer,
        VALIDATION_FILE,
        VALIDATION_OUTPUT,
    )

    metadata = {
        "tokenizer": str(TOKENIZER_PATH.relative_to(PROJECT_ROOT)),
        "vocabulary_size": vocab_size,
        "dtype": str(DTYPE),
        "bytes_per_token": np.dtype(DTYPE).itemsize,
        "special_tokens_added_automatically": False,
        "bos_added": False,
        "eos_added": False,
        "document_end_marker": DOCUMENT_END,
        "train": train_stats,
        "validation": validation_stats,
    }

    with METADATA_OUTPUT.open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print("\nEncoding completed!")

    print("\nTraining:")
    print(f"  Documents: {train_stats['documents']}")
    print(f"  Tokens: {train_stats['tokens']:,}")
    print(f"  Size: {train_stats['file_size_bytes']:,} bytes")

    print("\nValidation:")
    print(f"  Documents: {validation_stats['documents']}")
    print(f"  Tokens: {validation_stats['tokens']:,}")
    print(f"  Size: {validation_stats['file_size_bytes']:,} bytes")

    print(f"\nMetadata saved to: {METADATA_OUTPUT}")


if __name__ == "__main__":
    main()