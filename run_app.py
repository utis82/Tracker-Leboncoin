#!/usr/bin/env python3
"""Launcher script for Moto Leboncoin Analyzer.

This script ensures the Firecrawl API key is available before
starting the Dash application. It prompts the user for the key
once and stores it in a config file located in the user's home
directory so packaged executables can run without editing files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


CONFIG_DIR = Path.home() / ".moto_leboncoin"
CONFIG_FILE = CONFIG_DIR / "firecrawl_key.json"


def load_key_from_config() -> str | None:
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            key = data.get("firecrawl_api_key")
            if key:
                return key.strip()
        except (json.JSONDecodeError, OSError):
            pass
    return None


def save_key_to_config(key: str) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"firecrawl_api_key": key.strip()}
    CONFIG_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def prompt_for_key() -> str:
    print("\n🔥 Firecrawl API key requise pour lancer l'application.")
    print("Vous pouvez la générer sur https://app.firecrawl.dev/")
    while True:
        key = input("Entrez votre clé Firecrawl: ").strip()
        if key:
            return key
        print("❌ Clé invalide, veuillez réessayer.\n")


def ensure_firecrawl_key(reset: bool = False) -> str:
    env_key = os.getenv("FIRECRAWL_API_KEY")
    if env_key and not reset:
        return env_key
    if not reset:
        config_key = load_key_from_config()
        if config_key:
            os.environ["FIRECRAWL_API_KEY"] = config_key
            return config_key
    key = prompt_for_key()
    save_key_to_config(key)
    os.environ["FIRECRAWL_API_KEY"] = key
    return key


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lance l'application Moto Leboncoin Analyzer avec gestion de la clé Firecrawl."
    )
    parser.add_argument(
        "--reset-key",
        action="store_true",
        help="force la saisie d'une nouvelle clé Firecrawl",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="port HTTP de l'application (défaut: 8050)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=True,
        help="Active le mode debug Dash (défaut: activé)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_firecrawl_key(reset=args.reset_key)

    from app import app  # Import tardif pour éviter charges inutiles

    print("🚀 Lancement de l'application Moto Leboncoin Analyzer")
    print(f"📍 Accédez à l'application sur: http://localhost:{args.port}")
    app.run_server(debug=args.debug, port=args.port)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nArrêt demandé par l'utilisateur.")
        sys.exit(0)
