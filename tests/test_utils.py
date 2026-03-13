from pathlib import Path

# Shared Constants
TEST_DATA_DIR = Path("tests/test_data")
VALID_EXTS = {".dcm", ".png", ".jpg", ".jpeg"}
TEST_IMAGES = [p for p in TEST_DATA_DIR.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]