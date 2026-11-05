# data/prepare_dataset.py

from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import random
import re


# ==================================================
# 1. CONFIGURATION
# ==================================================

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")

RANDOM_SEED = 42
VALIDATION_RATIO = 0.10

MAX_FILE_SIZE = 1 * 1024 * 1024  # 1 MB
MIN_CHARACTERS = 50

CATEGORIES = {
    "python": "python",
    "cpp": "cpp",
    "csharp": "csharp",
    "java": "java",
    "documentation": "unknown",
    "dialogues": "unknown",
}

# Расширения исходного кода
CODE_EXTENSIONS = {
    ".py", ".cpp", ".cc", ".cxx",
    ".c", ".h", ".hpp", ".hh", ".hxx",
    ".cs", ".java",
}

# Текстовые документы и документация
DOCUMENTATION_EXTENSIONS = {
    ".md", ".rst", ".txt",
}

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    ".pytest_cache",
    ".idea",
    ".vscode",
    "vendor",
    "third_party",
    "third-party",
    "external",
    "bin",
    "obj",
}

# Эти файлы сохраняются в исходных репозиториях,
# но не попадают в обучающий текст.
EXCLUDED_FILE_NAMES = {
    "license",
    "license.md",
    "license.txt",
    "copying",
    "copying.md",
    "notice",
    "notice.md",
}

# Известные лицензии наших источников.
# Неизвестную лицензию не будем угадывать.
SOURCE_LICENSES = {
    "python/TheAlgorithms-Python": "MIT",
    "cpp/TheAlgorithms-C-Plus-Plus": "MIT",
    "java/TheAlgorithms-Java": "MIT",
    "csharp/TheAlgorithms-C-Sharp": "GPL-3.0",
}


# ==================================================
# 2. SPECIAL TOKENS
# ==================================================

LANGUAGE_TOKENS = {
    "python": "<|language_python|>",
    "cpp": "<|language_cpp|>",
    "csharp": "<|language_csharp|>",
    "java": "<|language_java|>",
    "unknown": "<|language_unknown|>",
}

TYPE_TOKENS = {
    "code": "<|type_code|>",
    "documentation": "<|type_documentation|>",
    "dialogue": "<|type_dialogue|>",
}

DOCUMENT_START = "<|document_start|>"
DOCUMENT_END = "<|document_end|>"


# ==================================================
# 3. HELPERS
# ==================================================

def sha256(text: str) -> str:
    """Вычисляет SHA-256 содержимого документа."""
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def get_source_id(category: str, file_path: Path) -> str:
    """
    Определяет источник документа.

    Например:
    cpp/Repository/algorithms/sort.cpp
    -> cpp/Repository
    """

    category_dir = RAW_DIR / category
    relative_path = file_path.relative_to(category_dir)

    if len(relative_path.parts) > 1:
        repository = relative_path.parts[0]
    else:
        repository = "root"

    return f"{category}/{repository}"


def get_license(source_id: str) -> str:
    """Возвращает известную лицензию источника."""
    return SOURCE_LICENSES.get(source_id, "UNKNOWN")


def should_exclude(file_path: Path, category_dir: Path) -> bool:
    """Проверяет, нужно ли пропустить файл."""

    relative_path = file_path.relative_to(category_dir)

    if any(
        part.lower() in EXCLUDED_DIRS
        for part in relative_path.parts[:-1]
    ):
        return True

    if file_path.name.lower() in EXCLUDED_FILE_NAMES:
        return True

    if file_path.suffix.lower() not in (
        CODE_EXTENSIONS | DOCUMENTATION_EXTENSIONS
    ):
        return True

    try:
        if file_path.stat().st_size > MAX_FILE_SIZE:
            return True
    except OSError:
        return True

    return False


def classify_document(
    category: str,
    file_path: Path
) -> tuple[str, str]:
    """
    Определяет язык и тип документа.

    Возвращает:
    (language, document_type)
    """

    extension = file_path.suffix.lower()

    # Диалоги в будущем будут обрабатываться отдельно.
    if category == "dialogues":
        return "unknown", "dialogue"

    # Документация может относиться к конкретному языку.
    language = CATEGORIES[category]

    if extension in DOCUMENTATION_EXTENSIONS:
        return language, "documentation"

    if extension in CODE_EXTENSIONS:
        return language, "code"

    return "unknown", "documentation"


# ==================================================
# 4. COLLECT DOCUMENTS
# ==================================================

def collect_documents():
    documents = []

    stats = {
        "found": 0,
        "excluded": 0,
        "too_small": 0,
        "unreadable": 0,
        "duplicates": 0,
    }

    # Дубликаты удаляем только внутри одного источника.
    # Так мы не потеряем сведения о разных лицензиях.
    seen_hashes = set()

    for category in CATEGORIES:
        category_dir = RAW_DIR / category

        if not category_dir.exists():
            continue

        print(f"Searching category: {category}")

        for file_path in category_dir.rglob("*"):
            if not file_path.is_file():
                continue

            if should_exclude(file_path, category_dir):
                stats["excluded"] += 1
                continue

            stats["found"] += 1

            try:
                content = file_path.read_text(
                    encoding="utf-8-sig"
                )
            except (UnicodeDecodeError, OSError):
                stats["unreadable"] += 1
                continue

            content = content.strip()

            if len(content) < MIN_CHARACTERS:
                stats["too_small"] += 1
                continue

            language, document_type = classify_document(
                category,
                file_path
            )

            source_id = get_source_id(
                category,
                file_path
            )

            content_hash = sha256(content)

            duplicate_key = (
                source_id,
                language,
                document_type,
                content_hash,
            )

            if duplicate_key in seen_hashes:
                stats["duplicates"] += 1
                continue

            seen_hashes.add(duplicate_key)

            documents.append({
                "content": content,
                "language": language,
                "type": document_type,
                "source": source_id,
                "license": get_license(source_id),
                "path": str(file_path),
            })

    return documents, stats


# ==================================================
# 5. TRAIN / VALIDATION SPLIT
# ==================================================

def split_documents(documents):
    rng = random.Random(RANDOM_SEED)

    train = []
    validation = []
    used_methods = set()

    languages = sorted({
        doc["language"] for doc in documents
    })

    for language in languages:
        language_docs = [
            doc for doc in documents
            if doc["language"] == language
        ]

        sources = sorted({
            doc["source"]
            for doc in language_docs
            if not doc["source"].endswith("/root")
        })

        # При наличии нескольких источников
        # отделяем целые репозитории.
        if len(sources) >= 2:
            rng.shuffle(sources)

            validation_count = max(
                1,
                round(len(sources) * VALIDATION_RATIO)
            )

            validation_sources = set(
                sources[:validation_count]
            )

            for doc in language_docs:
                if doc["source"] in validation_sources:
                    validation.append(doc)
                else:
                    train.append(doc)

            used_methods.add("source_level")

        else:
            # Пока для языка есть только один источник,
            # используем временное разделение по файлам.
            rng.shuffle(language_docs)

            validation_count = (
                max(1, round(
                    len(language_docs) * VALIDATION_RATIO
                ))
                if len(language_docs) >= 2
                else 0
            )

            validation.extend(
                language_docs[:validation_count]
            )

            train.extend(
                language_docs[validation_count:]
            )

            used_methods.add("file_level_single_source")

    rng.shuffle(train)
    rng.shuffle(validation)

    if len(used_methods) == 1:
        split_method = next(iter(used_methods))
    else:
        split_method = "mixed"

    return train, validation, split_method


# ==================================================
# 6. SAVE DOCUMENTS
# ==================================================

def format_document(doc: dict) -> str:
    """Добавляет специальные маркеры к документу."""

    language_token = LANGUAGE_TOKENS.get(
        doc["language"],
        LANGUAGE_TOKENS["unknown"]
    )

    type_token = TYPE_TOKENS[doc["type"]]

    return (
        f"{DOCUMENT_START}\n"
        f"{language_token}\n"
        f"{type_token}\n\n"
        f"{doc['content']}\n"
        f"{DOCUMENT_END}\n"
    )


def save_text_dataset(
    documents: list,
    output_path: Path
):
    """Сохраняет размеченные документы в один текстовый файл."""

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:
        for doc in documents:
            file.write(format_document(doc))
            file.write("\n")


def save_metadata(
    documents: list,
    output_path: Path
):
    """
    Сохраняет метаданные в JSONL.

    Одна строка = один документ.
    """

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:
        for doc in documents:
            metadata = {
                "language": doc["language"],
                "type": doc["type"],
                "source": doc["source"],
                "license": doc["license"],
                "path": doc["path"],
            }

            file.write(
                json.dumps(metadata, ensure_ascii=False)
                + "\n"
            )


# ==================================================
# 7. STATISTICS
# ==================================================

def count_field(documents: list, field: str) -> dict:
    return dict(Counter(
        doc[field] for doc in documents
    ))


def count_licenses(documents: list) -> dict:
    result = defaultdict(Counter)

    for doc in documents:
        result[doc["source"]][doc["license"]] += 1

    return {
        source: dict(licenses)
        for source, licenses in result.items()
    }


# ==================================================
# 8. MAIN
# ==================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Searching for files...")

    documents, stats = collect_documents()

    print(f"Found: {stats['found']}")
    print(f"Unique: {len(documents)}")
    print(f"Duplicates: {stats['duplicates']}")
    print(f"Unreadable: {stats['unreadable']}")

    if len(documents) < 2:
        raise RuntimeError(
            "Not enough documents to create a dataset."
        )

    train, validation, split_method = split_documents(
        documents
    )

    # Текстовые корпуса
    save_text_dataset(
        train,
        OUTPUT_DIR / "train.txt"
    )

    save_text_dataset(
        validation,
        OUTPUT_DIR / "validation.txt"
    )

    # Метаданные
    save_metadata(
        train,
        OUTPUT_DIR / "train_metadata.jsonl"
    )

    save_metadata(
        validation,
        OUTPUT_DIR / "validation_metadata.jsonl"
    )

    manifest = {
        "total_files": stats["found"],
        "unique_files": len(documents),

        "training_files": len(train),
        "validation_files": len(validation),

        "duplicates": stats["duplicates"],
        "unreadable": stats["unreadable"],
        "too_small": stats["too_small"],
        "excluded": stats["excluded"],

        "random_seed": RANDOM_SEED,
        "validation_ratio": VALIDATION_RATIO,
        "split_method": split_method,

        "train_languages": count_field(
            train, "language"
        ),

        "validation_languages": count_field(
            validation, "language"
        ),

        "train_types": count_field(
            train, "type"
        ),

        "validation_types": count_field(
            validation, "type"
        ),

        "train_sources": count_field(
            train, "source"
        ),

        "validation_sources": count_field(
            validation, "source"
        ),

        "licenses": count_licenses(documents),

        "special_tokens": [
            DOCUMENT_START,
            DOCUMENT_END,
            *LANGUAGE_TOKENS.values(),
            *TYPE_TOKENS.values(),
        ],
    }

    with (
        OUTPUT_DIR / "manifest.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            manifest,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nDataset prepared!")

    print(json.dumps(
        manifest,
        indent=4,
        ensure_ascii=False
    ))


if __name__ == "__main__":
    main()