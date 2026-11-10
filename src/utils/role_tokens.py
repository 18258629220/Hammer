"""Register and persist the CRIM vocabulary contract without model imports."""

import hashlib
import json
from pathlib import Path

from .experiment import write_provenance


ROLE_TOKENS = {
    "contact": "[CONTACT]",
    "effect": "[EFFECT]",
    "exclude": "[EXCLUDE]",
}


def _role_ids(tokenizer, vocabulary):
    ids = {}
    for role, token in ROLE_TOKENS.items():
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"The {role} token must encode to exactly one token ID")
        token_id = encoded[0]
        if (not isinstance(token_id, int) or isinstance(token_id, bool)
                or not 0 <= token_id < len(tokenizer)
                or vocabulary.get(token) != token_id):
            raise ValueError(f"The {role} token is missing or has an invalid vocabulary ID")
        ids[role] = token_id
    if len(set(ids.values())) != len(ROLE_TOKENS):
        raise ValueError("CONTACT, EFFECT and EXCLUDE must have distinct token IDs")
    return ids


def register_role_tokens(tokenizer):
    """Add the three roles while retaining Qwen's existing special tokens."""
    tokenizer.add_special_tokens(
        {"additional_special_tokens": list(ROLE_TOKENS.values())},
        replace_additional_special_tokens=False,
    )
    return _role_ids(tokenizer, tokenizer.get_vocab())


def tokenizer_record(tokenizer, base_version):
    """Describe an already registered vocabulary without adding tokens again."""
    if not isinstance(base_version, str) or not base_version:
        raise ValueError("The tokenizer base model version must be a nonempty string")
    vocabulary = tokenizer.get_vocab()
    if not isinstance(vocabulary, dict) or any(
        not isinstance(token, str) or not isinstance(index, int)
        or isinstance(index, bool) or index < 0
        for token, index in vocabulary.items()
    ):
        raise ValueError("The tokenizer vocabulary must map strings to nonnegative integer IDs")
    encoded = json.dumps(
        vocabulary, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return {
        "schema_version": 1,
        "role_token_ids": _role_ids(tokenizer, vocabulary),
        "vocab_size": len(tokenizer),
        "base_version": base_version,
        "vocab_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def validate_role_record(tokenizer, record, base_version):
    """Reject any drift in the saved role IDs, vocabulary, or base version."""
    expected = tokenizer_record(tokenizer, base_version)
    if not isinstance(record, dict) or set(record) != set(expected):
        raise ValueError("The saved role token record has missing or unsupported fields")
    mismatches = [
        name for name, value in expected.items()
        if type(record[name]) is not type(value) or record[name] != value
    ]
    role_ids = record["role_token_ids"]
    if isinstance(role_ids, dict) and any(type(value) is not int for value in role_ids.values()):
        mismatches.append("role_token_ids")
    if mismatches:
        raise ValueError("Saved role tokenizer contract mismatch: " + ", ".join(sorted(set(mismatches))))
    return expected


def save_role_processor(processor, directory, base_version):
    """Save the native processor and atomically write its role vocabulary record."""
    record = tokenizer_record(processor.tokenizer, base_version)
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(str(directory))
    write_provenance(directory / "role_tokens.json", record)
    return record
