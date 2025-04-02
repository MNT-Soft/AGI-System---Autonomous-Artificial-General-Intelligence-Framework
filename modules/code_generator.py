import subprocess
import tempfile
from typing import Tuple, Optional
from config.settings import settings

class CodeExecutor:
    TEMPLATES = {
        "python": {
            "analysis": "import pandas as pd\ndata = pd.read_csv('{filename}')",
            "optimization": "def optimize():\n    return sum(range({size}))"
        },
        "cpp": {
            "calculation": "#include <iostream>\nint main() {{\n    int sum=0;\n    for(int i=0;i<{size};++i) sum+=i;\n    std::cout << sum;\n}}"
        }
    }

    def execute(self, task_type: str, lang: str = "python", **params) -> Tuple[bool, str]:
        if not settings.ENABLE_CODE_EXEC:
            return False, "Code execution disabled"
            
        template = self.TEMPLATES.get(lang, {}).get(task_type)
        if not template:
            return False, "Unsupported task/language"
            
        code = template.format(**params)
        
        with tempfile.NamedTemporaryFile(mode='w+', suffix=f'.{lang}', delete=False) as f:
            f.write(code)
            filepath = f.name
            
        try:
            if lang == "python":
                result = subprocess.run(
                    ["python", filepath],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
            elif lang == "cpp":
                subprocess.run(
                    ["g++", filepath, "-o", filepath[:-4]],
                    check=True
                )
                result = subprocess.run(
                    [f"./{filepath[:-4]}"],
                    capture_output=True,
                    text=True
                )
            return True, result.stdout
        except subprocess.SubprocessError as e:
            return False, str(e)
        finally:
            subprocess.run(["rm", filepath], capture_output=True)