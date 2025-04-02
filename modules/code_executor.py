import subprocess
import tempfile
import os
import time
from typing import Tuple, Dict, Optional
from config import config
import logging
from threading import Lock

logger = logging.getLogger(__name__)

class CodeExecutor:
    SUPPORTED_LANGUAGES = {
        "python": {
            "extension": ".py",
            "command": ["python", "{filepath}"]
        },
        "cpp": {
            "extension": ".cpp",
            "compile": ["g++", "{filepath}", "-o", "{executable}"],
            "command": ["./{executable}"]
        }
    }

    TEMPLATES = {
        "analysis": {
            "python": "import pandas as pd\ndata = pd.read_csv('{filename}')",
            "cpp": "#include <iostream>\n#include <vector>\n// Analysis code"
        },
        "optimization": {
            "python": "def optimize():\n    return sum(range({size}))",
            "cpp": "#include <iostream>\nint main() {{\n    int sum=0;\n    for(int i=0;i<{size};++i) sum+=i;\n    std::cout << sum;\n}}"
        }
    }

    def __init__(self):
        self.lock = Lock()

    def execute(self, task_type: str, lang: str = "python", params: Optional[Dict] = None) -> Tuple[bool, str]:
        """Безопасное выполнение кода"""
        if not config.ENABLE_CODE_EXEC:
            return False, "Выполнение кода отключено в настройках"

        if not params:
            params = {}

        # Валидация входных параметров
        if not isinstance(task_type, str) or not isinstance(lang, str):
            return False, "Неверный тип параметров"
            
        if lang not in self.SUPPORTED_LANGUAGES:
            return False, f"Неподдерживаемый язык: {lang}"

        with self.lock:
            try:
                # Создание временного файла
                with tempfile.NamedTemporaryFile(
                    mode='w+', 
                    suffix=self.SUPPORTED_LANGUAGES[lang]["extension"], 
                    delete=False,
                    encoding='utf-8'
                ) as f:
                    code = self._generate_code(task_type, lang, params)
                    if not code:
                        return False, "Не удалось сгенерировать код"
                        
                    f.write(code)
                    filepath = f.name
                    executable = filepath[:-4] if lang == "cpp" else None

                # Компиляция (для C++)
                if lang == "cpp":
                    compile_cmd = [
                        part.format(filepath=filepath, executable=executable) 
                        for part in self.SUPPORTED_LANGUAGES[lang]["compile"]
                    ]
                    self._run_command(compile_cmd, "Компиляция")

                # Выполнение
                exec_cmd = [
                    part.format(filepath=filepath, executable=executable) 
                    for part in self.SUPPORTED_LANGUAGES[lang]["command"]
                ]
                success, output = self._run_command(exec_cmd, "Выполнение")

                return success, output

            except Exception as e:
                logger.error("Ошибка выполнения кода: %s", str(e))
                return False, f"Ошибка выполнения: {str(e)}"
            finally:
                # Очистка
                self._cleanup(filepath, executable if lang == "cpp" else None)

    def _generate_code(self, task_type: str, lang: str, params: Dict) -> str:
        """Генерация кода из шаблона с валидацией"""
        template = self.TEMPLATES.get(task_type, {}).get(lang)
        if not template:
            raise ValueError(f"Неподдерживаемый тип задачи или язык: {task_type}/{lang}")

        # Валидация параметров
        for key, value in params.items():
            if not isinstance(key, str) or not isinstance(value, (str, int, float)):
                raise ValueError("Некорректные параметры кода")

        return template.format(**params)

    def _run_command(self, command: list, stage: str) -> Tuple[bool, str]:
        """Безопасное выполнение команды"""
        try:
            start_time = time.time()
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=config.MAX_CODE_EXEC_TIME,
                check=True
            )
            exec_time = time.time() - start_time
            
            logger.info("%s успешно завершена за %.2f сек", stage, exec_time)
            return True, f"{result.stdout}\nВремя выполнения: {exec_time:.2f} сек"
            
        except subprocess.TimeoutExpired:
            logger.warning("%s превышено время ожидания", stage)
            return False, f"{stage}: превышено время выполнения"
        except subprocess.CalledProcessError as e:
            logger.warning("%s завершилась с ошибкой: %s", stage, e.stderr)
            return False, f"{stage} ошибка:\n{e.stderr}"
        except Exception as e:
            logger.error("Неожиданная ошибка при %s: %s", stage, str(e))
            return False, f"Неожиданная ошибка: {str(e)}"

    def _cleanup(self, filepath: str, executable: Optional[str] = None):
        """Очистка временных файлов"""
        try:
            if filepath and os.path.exists(filepath):
                os.unlink(filepath)
            if executable and os.path.exists(executable):
                os.unlink(executable)
        except Exception as e:
            logger.error("Ошибка очистки: %s", str(e))
