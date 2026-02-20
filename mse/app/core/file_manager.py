import json
from pathlib import Path
from typing import Dict, Any, Optional


class StatusManager:

    def __init__(self, status_file: str = "status.json"):

        self.status_file = Path(status_file)

    def read_status(self) -> Dict[str, Any]:

        if not self.status_file.exists():
            return {
                "current_model": 0,
                "total_models": 0,
                "status": "idle",
                "current_config": None
            }

        try:
            with open(self.status_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Ошибка {self.status_file}")
            return {}
        except Exception as e:
            print(f"Ошибка: {e}")
            return {}

    def write_status(self, data: Dict[str, Any]) -> bool:
        try:
            with open(self.status_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Ошибка при записи статуса: {e}")
            return False

    def update_status(self, **kwargs) -> bool:

        current = self.read_status()
        current.update(kwargs)
        return self.write_status(current)



if __name__ == "__main__":
    manager = StatusManager()

    status = manager.read_status()
    print("Текущий статус:", status)

    manager.update_status(
        status="running",
        current_model=1,
        total_models=5
    )
    print("Статус обновлен")
    new_status = manager.read_status()
    print("Новый статус:", new_status)