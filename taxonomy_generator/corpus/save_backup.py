import shutil
from pathlib import Path

from taxonomy_generator.utils.utils import unique_file

SOURCE_PATH = Path("data/ai_safety_corpus.csv")
BACKUP_PATH = Path().resolve().parent.parent / "_archive"


def save_backup():
    """
    Creates a backup of the AI safety corpus by copying the source file
    to the backup directory with a timestamp in the filename.
    """
    BACKUP_PATH.mkdir(parents=True, exist_ok=True)

    backup_file_path = BACKUP_PATH / unique_file("ai_safety_corpus_{}.csv")

    shutil.copy2(SOURCE_PATH, backup_file_path)

    return backup_file_path


if __name__ == "__main__":
    backup_path = save_backup()
    print(f"Backup created at: {backup_path}")
