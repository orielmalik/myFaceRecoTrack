import os
from pathlib import Path

from dotenv import load_dotenv


def find_and_load_env(key_name: str, filename: str = "k.env"):
    root = Path(__file__).resolve().parents[1]
    env_path = root / filename
    if not env_path.exists():
        raise FileNotFoundError(f".env not found at {env_path}")

    load_dotenv(env_path)

    value = os.getenv(key_name)

    if not value:
        raise ValueError(f"{key_name} not found in env")

    return value.strip()
