#Task 1.2.3

from pathlib import Path

def find_latest_train_folder(runs_dir: str = "runs/detect"):
    runs_path = Path(runs_dir)

    if not runs_path.exists():
        print(f"Директории {runs_dir} не существует")
        return None

    folders = [f for f in runs_path.iterdir() if f.is_dir()]

    if not folders:
        print(f"В Директории {runs_dir} нет поддиректорий")
        return None
    latest_folder = max(folders, key=lambda f: f.stat().st_ctime)

    return str(latest_folder)


if __name__ == "__main__":
    #Path("runs/detect/train10").mkdir(parents=True, exist_ok=True)
    #Path("runs/detect/train11").mkdir(parents=True, exist_ok=True)
    #Path("runs/detect/train12").mkdir(parents=True, exist_ok=True)

    latest = find_latest_train_folder()
    print(f"Последняя созданная директория: {latest}")
