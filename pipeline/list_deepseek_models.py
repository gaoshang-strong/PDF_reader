#!/usr/bin/env python3
"""List models available on the DeepSeek API."""

import os
import sys


def main() -> None:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("Error: DEEPSEEK_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    from openai import OpenAI

    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    client = OpenAI(api_key=api_key, base_url=base_url)

    try:
        models = client.models.list()
    except Exception as exc:
        print(f"Error fetching models: {exc}", file=sys.stderr)
        sys.exit(1)

    ids = sorted(m.id for m in models.data)
    if not ids:
        print("No models returned.")
        return

    print(f"Available DeepSeek models ({base_url}):")
    for mid in ids:
        print(f"  {mid}")


if __name__ == "__main__":
    main()
