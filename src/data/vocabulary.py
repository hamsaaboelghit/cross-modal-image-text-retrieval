
"""
Vocabulary loading and validation utilities for VisionMatch AI.

The vocabulary format is a JSON dictionary:
    {
        "<PAD>": 0,
        "<UNK>": 1,
        "<BOS>": 2,
        "<EOS>": 3,
        ...
    }

PAD must have ID 0 because TextEncoder uses PAD_IDX = 0.
"""

import json
from pathlib import Path


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"

SPECIAL_TOKENS = (
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
)

PAD_IDX = 0


def validate_vocabulary(word_to_idx):
    """
    Validate a word-to-index vocabulary.

    Returns:
        The validated dictionary.
    """

    if not isinstance(word_to_idx, dict):
        raise TypeError(
            "Vocabulary must be a dictionary."
        )

    if not word_to_idx:
        raise ValueError(
            "Vocabulary cannot be empty."
        )

    for token in SPECIAL_TOKENS:
        if token not in word_to_idx:
            raise ValueError(
                f"Vocabulary is missing special token: {token}"
            )

    ids = list(word_to_idx.values())

    for token_id in ids:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError(
                "Every vocabulary ID must be an integer."
            )

        if token_id < 0:
            raise ValueError(
                "Vocabulary IDs cannot be negative."
            )

    if len(set(ids)) != len(ids):
        raise ValueError(
            "Vocabulary IDs must be unique."
        )

    expected_ids = set(range(len(word_to_idx)))

    if set(ids) != expected_ids:
        raise ValueError(
            "Vocabulary IDs must be contiguous, starting from zero."
        )

    for expected_id, token in enumerate(SPECIAL_TOKENS):
        if word_to_idx[token] != expected_id:
            raise ValueError(
                f"{token} must have ID {expected_id}, "
                f"but found {word_to_idx[token]}."
            )

    if word_to_idx[PAD_TOKEN] != PAD_IDX:
        raise ValueError(
            f"PAD token must have ID {PAD_IDX}."
        )

    return word_to_idx


def load_vocabulary(vocabulary_path):
    """
    Load and validate a vocabulary JSON file.

    Args:
        vocabulary_path: Path to a vocabulary JSON file.

    Returns:
        Validated word_to_idx dictionary.
    """

    vocabulary_path = Path(vocabulary_path)

    if not vocabulary_path.exists():
        raise FileNotFoundError(
            f"Vocabulary file not found: {vocabulary_path}"
        )

    if not vocabulary_path.is_file():
        raise ValueError(
            f"Vocabulary path is not a file: {vocabulary_path}"
        )

    try:
        with vocabulary_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            word_to_idx = json.load(file)

    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid vocabulary JSON: {error}"
        ) from error

    return validate_vocabulary(word_to_idx)


def get_vocabulary_size(word_to_idx):
    """
    Return vocabulary size after validating the mapping.
    """

    validate_vocabulary(word_to_idx)

    return len(word_to_idx)


def get_special_token_ids(word_to_idx):
    """
    Return the IDs of the four required special tokens.
    """

    validate_vocabulary(word_to_idx)

    return {
        "pad_id": word_to_idx[PAD_TOKEN],
        "unk_id": word_to_idx[UNK_TOKEN],
        "bos_id": word_to_idx[BOS_TOKEN],
        "eos_id": word_to_idx[EOS_TOKEN],
    }


def _run_vocabulary_tests():
    import tempfile

    print("Vocabulary Utility Tests")
    print("-" * 55)

    valid_vocabulary = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<BOS>": 2,
        "<EOS>": 3,
        "a": 4,
        "cat": 5,
        "runs": 6,
    }

    # Test 1: Valid vocabulary.
    validated = validate_vocabulary(
        valid_vocabulary.copy()
    )

    assert validated == valid_vocabulary

    print("Valid vocabulary: PASS")

    # Test 2: Vocabulary size.
    vocab_size = get_vocabulary_size(
        valid_vocabulary
    )

    assert vocab_size == 7

    print("Vocabulary size: PASS")

    # Test 3: Special token IDs.
    special_ids = get_special_token_ids(
        valid_vocabulary
    )

    assert special_ids == {
        "pad_id": 0,
        "unk_id": 1,
        "bos_id": 2,
        "eos_id": 3,
    }

    print("Special token IDs: PASS")

    # Test 4: Save and load JSON.
    with tempfile.TemporaryDirectory() as temp_dir:
        vocabulary_path = (
            Path(temp_dir) / "test_vocab.json"
        )

        with vocabulary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                valid_vocabulary,
                file,
                ensure_ascii=False,
                indent=4,
            )

        loaded_vocabulary = load_vocabulary(
            vocabulary_path
        )

        assert loaded_vocabulary == valid_vocabulary

    print("JSON vocabulary loading: PASS")

    # Test 5: Missing special token.
    invalid_missing = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<BOS>": 2,
        "cat": 3,
    }

    try:
        validate_vocabulary(invalid_missing)
        raise AssertionError(
            "Missing special token was accepted."
        )
    except ValueError:
        print("Missing special token validation: PASS")

    # Test 6: Incorrect PAD ID.
    invalid_pad = {
        "<PAD>": 1,
        "<UNK>": 0,
        "<BOS>": 2,
        "<EOS>": 3,
    }

    try:
        validate_vocabulary(invalid_pad)
        raise AssertionError(
            "Incorrect PAD ID was accepted."
        )
    except ValueError:
        print("PAD ID validation: PASS")

    # Test 7: Duplicate IDs.
    invalid_duplicate = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<BOS>": 2,
        "<EOS>": 3,
        "cat": 3,
    }

    try:
        validate_vocabulary(invalid_duplicate)
        raise AssertionError(
            "Duplicate IDs were accepted."
        )
    except ValueError:
        print("Duplicate ID validation: PASS")

    # Test 8: Non-contiguous IDs.
    invalid_gap = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<BOS>": 2,
        "<EOS>": 3,
        "cat": 5,
    }

    try:
        validate_vocabulary(invalid_gap)
        raise AssertionError(
            "Non-contiguous IDs were accepted."
        )
    except ValueError:
        print("Contiguous ID validation: PASS")

    # Test 9: Invalid ID type.
    invalid_type = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<BOS>": 2,
        "<EOS>": 3,
        "cat": "4",
    }

    try:
        validate_vocabulary(invalid_type)
        raise AssertionError(
            "Non-integer ID was accepted."
        )
    except TypeError:
        print("ID type validation: PASS")

    print("-" * 55)
    print("All vocabulary tests passed!")


if __name__ == "__main__":
    _run_vocabulary_tests()
