import random
import shutil
from pathlib import Path
from typing import Iterable, List

from config import CONFIG
from utils import set_global_seed

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _iter_class_dirs(base_dir: Path) -> Iterable[Path]:
    for item in base_dir.iterdir():
        if not item.is_dir():
            continue
        if item.name in {CONFIG.dataset.train_dir.name, CONFIG.dataset.test_dir.name}:
            continue
        yield item


def _collect_images(class_dir: Path) -> List[Path]:
    return [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]


def _clear_directory(directory: Path) -> None:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


def _copy_files(files: Iterable[Path], destination_root: Path, class_name: str) -> None:
    class_dest = destination_root / class_name
    class_dest.mkdir(parents=True, exist_ok=True)
    for src in files:
        shutil.copy2(src, class_dest / src.name)


def main() -> None:
    set_global_seed(CONFIG.seed)

    project_root = _project_root()

    base_dir = CONFIG.dataset.base_dir
    train_root = CONFIG.dataset.train_dir
    test_root = CONFIG.dataset.test_dir

    base_dir = base_dir if base_dir.is_absolute() else project_root / base_dir
    train_root = train_root if train_root.is_absolute() else project_root / train_root
    test_root = test_root if test_root.is_absolute() else project_root / test_root

    if not base_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {base_dir}")

    _clear_directory(train_root)
    _clear_directory(test_root)

    rng = random.Random(CONFIG.seed)

    for class_dir in _iter_class_dirs(base_dir):
        images = _collect_images(class_dir)
        if not images:
            print(f"[WARN] No images found for class '{class_dir.name}', skipping.")
            continue

        shuffled = images[:]
        rng.shuffle(shuffled)

        test_count = int(len(shuffled) * CONFIG.dataset.test_split)
        if len(shuffled) > 1:
            test_count = max(1, min(test_count, len(shuffled) - 1))
        else:
            test_count = 1

        test_files = shuffled[:test_count]
        train_files = shuffled[test_count:]

        _copy_files(train_files, train_root, class_dir.name)
        _copy_files(test_files, test_root, class_dir.name)

        print(
            f"[INFO] Class '{class_dir.name}': {len(train_files)} training and {len(test_files)} testing images prepared."
        )

    print(f"[INFO] Train images stored in: {train_root}")
    print(f"[INFO] Test images stored in: {test_root}")


if __name__ == "__main__":
    main()
