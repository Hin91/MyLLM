from pathlib import Path
from tokenizers import Tokenizer


PROJECT_DIR = Path(__file__).resolve().parent.parent

TOKENIZER_PATH = PROJECT_DIR / "tokenizer" / "tokenizer.json"


def main():
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    test_cases = [
        "print('Hello, world!')",

        "def calculate_sum(a, b):\n    return a + b",

        "class MyClass:\n    pass",

        "x = [1, 2, 3, 4]",

        "def неизвестная_функция():\n    return 42",
    ]

    for text in test_cases:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)

        print("=" * 50)
        print("Original:", repr(text))
        print("IDs:", encoded.ids)
        print("Tokens:", encoded.tokens)
        print("Decoded:", repr(decoded))

        assert decoded == text, (
            f"Text reconstruction failed: {text!r}"
        )

    print("\nAll tests passed!")


if __name__ == "__main__":
    main()