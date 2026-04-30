"""
Simple token chunk inspector.

Use this when you want to see which raw control chunk belongs to one token id.
"""

from pathlib import Path
import argparse
import json

from tokenizer import get_token_chunk, load_tokenizer


def describe_token_chunk(token_id, tokenizer_path=None):
    """
    Return one token's chunk metadata from the saved tokenizer file.
    """

    if tokenizer_path is None:
        tokenizer_path = (
            Path(__file__).resolve().parent / "training_data" / "tokenizer_lookup.json"
        )

    tokenizer_state = load_tokenizer(tokenizer_path)
    return get_token_chunk(tokenizer_state, token_id)


def main():
    """
    Print one token chunk in JSON form.
    """

    parser = argparse.ArgumentParser(description="Inspect the raw chunk for one token id")
    parser.add_argument("token_id", type=int)
    parser.add_argument(
        "--tokenizer",
        default=str(Path(__file__).resolve().parent / "training_data" / "tokenizer_lookup.json"),
    )
    args = parser.parse_args()

    token_entry = describe_token_chunk(args.token_id, args.tokenizer)
    print(json.dumps(token_entry, indent=2))


if __name__ == "__main__":
    main()
