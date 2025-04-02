import subprocess
import tempfile
from typing import Tuple

class CodeGenerator:
    SUPPORTED_LANGUAGES = {
        'python': {'ext': '.py', 'command': 'python'},
        'cpp': {'ext': '.cpp', 'command': 'g++ -o {temp_file} {file} && ./{temp_file}'}
    }
    
    def generate(self, task: str, lang: str = 'python') -> Tuple[bool, str]:
        """Генерация кода по описанию задачи"""
        template = self._select_template(task, lang)
        if not template:
            return False, "Unsupported task/language combination"
            
        with tempfile.NamedTemporaryFile(suffix=self.SUPPORTED_LANGUAGES[lang]['ext'], delete=False) as f:
            f.write(template.encode('utf-8'))
            file_path = f.name
            
        try:
            cmd = self.SUPPORTED_LANGUAGES[lang]['command'].format(
                file=file_path,
                temp_file=file_path[:-4]
            )
            result = subprocess.run(cmd, shell=True, check=True, 
                                 capture_output=True, text=True, timeout=10)
            return True, result.stdout
        except subprocess.SubprocessError as e:
            return False, str(e)
            
    def _select_template(self, task: str, lang: str) -> str:
        """Выбор шаблона кода"""
        templates = {
            'python': {
                'optimize': "def optimize():\n    return sum(range(1000000))",
                'analyze': "import pandas as pd\ndata = pd.read_csv('data.csv')"
            },
            'cpp': {
                'optimize': "#include <iostream>\nint main() { int sum=0; for(int i=0;i<1000000;++i) sum+=i; std::cout << sum; }"
            }
        }
        return templates.get(lang, {}).get(task, "")