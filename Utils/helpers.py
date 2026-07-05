import json
import itertools
from pathlib import Path


def get_data_from_json(filename):
    current_dir = Path.cwd()
    file_path = None

    pure_filename = Path(filename).name

    while True:
        # אופציה 1: בדיקה אם הקובץ נמצא ישירות בתיקייה הנוכחית
        direct_path = current_dir / pure_filename
        if direct_path.is_file():
            file_path = direct_path
            break


        found_files = list(current_dir.rglob(pure_filename))
        if found_files:
            file_path = found_files[0]
            break

        if current_dir == current_dir.parent:
            raise FileNotFoundError(
                f"Fatal Error: The file '{pure_filename}' was not found directly or within "
                f"any subdirectories along the entire upward path to the root directory."
            )

        current_dir = current_dir.parent

    with open(file_path, "r", encoding="utf-8") as file:
        raw = json.load(file)

    normalized = {
        k: v if isinstance(v, list) else [v]
        for k, v in raw.items()
    }

    keys = list(normalized.keys())
    values = list(normalized.values())

    cases = [
        dict(zip(keys, combo))
        for combo in itertools.product(*values)
    ]

    return cases, raw