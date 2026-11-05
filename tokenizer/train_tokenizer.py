from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer


# ==================================================
# Configuration
# ==================================================

TRAIN_FILE = Path("data/processed/train.txt")
OUTPUT_DIR = Path("tokenizer")
OUTPUT_FILE = OUTPUT_DIR / "tokenizer.json"

VOCAB_SIZE = 8000
MIN_FREQUENCY = 2


# ==================================================
# Special tokens
# ==================================================

SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",

    "<|document_start|>",
    "<|document_end|>",

    "<|language_python|>",
    "<|language_cpp|>",
    "<|language_csharp|>",
    "<|language_java|>",
    "<|language_unknown|>",

    "<|type_code|>",
    "<|type_documentation|>",
    "<|type_dialogue|>",
]


# ==================================================
# Training
# ==================================================

def train_tokenizer():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAIN_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    tokenizer = Tokenizer(
        BPE(unk_token="<unk>")
    )

    tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space=False
    )

    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )

    print("Training tokenizer...")
    print(f"Dataset: {TRAIN_FILE}")

    # ВАЖНО:
    # Используем только train.txt.
    # validation.txt не должен участвовать
    # в обучении токенизатора.
    tokenizer.train(
        files=[str(TRAIN_FILE)],
        trainer=trainer
    )

    tokenizer.save(str(OUTPUT_FILE))

    print("\nTokenizer trained!")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Saved to: {OUTPUT_FILE}")

    return tokenizer


# ==================================================
# Special token tests
# ==================================================

def test_special_tokens(tokenizer):
    print("\nTesting special tokens...")

    for token in SPECIAL_TOKENS:
        token_id = tokenizer.token_to_id(token)

        if token_id is None:
            raise AssertionError(
                f"Special token missing: {token}"
            )

        encoded = tokenizer.encode(token)

        if len(encoded.ids) != 1:
            raise AssertionError(
                f"Token is not atomic: {token}"
            )

        print(f"{token:30} -> ID {token_id}")

    print("\nAll special tokens passed!")


# ==================================================
# Round-trip test
# ==================================================

def test_round_trip(tokenizer):
    print("\nTesting text encoding/decoding...")

    examples = [
        "def hello():\n    return 'Hello world'",
        "int main() { return 0; }",
        "public class Main { }",
        "using System;\nclass Program { }",
        "Привет, мир!",
        "<|document_start|>\n<|language_python|>",
    ]

    for text in examples:
        encoded = tokenizer.encode(text)

        decoded = tokenizer.decode(
            encoded.ids,
            skip_special_tokens=False
        )

        if decoded != text:
            raise AssertionError(
                f"Round-trip failed!\n"
                f"Original: {text!r}\n"
                f"Decoded: {decoded!r}"
            )

        print(f"Passed: {text[:40]!r}")

    print("\nAll round-trip tests passed!")


# ==================================================
# Main
# ==================================================

if __name__ == "__main__":
    tokenizer = train_tokenizer()

    test_special_tokens(tokenizer)
    test_round_trip(tokenizer)