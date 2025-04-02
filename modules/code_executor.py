import subprocess
import tempfile
from typing import Tuple, Dict, Optional
from config import config
import time

class CodeExecutor:
    SUPPORTED_LANGUAGES = {
        "python": {
            "extension": ".py",
            "command": "python {filepath}"
        },
        "cpp": {
            "extension": ".cpp",
            "compile": "g++ {filepath} -o {executable}",
            "command": "./{executable}"
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

    def execute(self, task_type: str, lang: str = "python", params: Optional[Dict] = None) -> Tuple[bool, str]:
        """Выполнение сгенерированного кода"""
        if not params:
            params = {}
            
        # Получение шаблона
        template = self.TEMPLATES.get(task_type, {}).get(lang)
        if not template:
            return False, f"Unsupported task type or language: {task_type}/{lang}"
        
        # Заполнение шаблона
        code = template.format(**params)
        
        # Создание временного файла
        with tempfile.NamedTemporaryFile(mode='w+', suffix=self.SUPPORTED_LANGUAGES[lang]["extension"], delete=False) as f:
            f.write(code)
            filepath = f.name
        
        try:
            # Компиляция (для C++)
            if lang == "cpp":
                executable = filepath[:-4]
                compile_cmd = self.SUPPORTED_LANGUAGES[lang]["compile"].format(
                    filepath=filepath,
                    executable=executable
                )
                subprocess.run(compile_cmd.split(), check=True, timeout=config.MAX_CODE_EXEC_TIME)
            
            # Выполнение
            exec_cmd = self.SUPPORTED_LANGUAGES[lang]["command"].format(
                filepath=filepath,
                executable=executable if lang == "cpp" else ""
            )
            
            start_time = time.time()
            result = subprocess.run(
                exec_cmd.split(),
                capture_output=True,
                text=True,
                timeout=config.MAX_CODE_EXEC_TIME
            )
            
            execution_time = time.time() - start_time
            output = f"Execution time: {execution_time:.2f}s\n{result.stdout}"
            
            return True, output
            
        except subprocess.SubprocessError as e:
            return False, str(e)
        finally:
            # Очистка
            subprocess.run(["rm", filepath], capture_output=True)
            if lang == "cpp":
                subprocess.run(["rm", executable], capture_output=True)