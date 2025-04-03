import ast
import os
import sys
from typing import Dict, Optional
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

class CodeEvolver:
    def __init__(self, target_file: str = "core/organism.py"):
        self.target_file = target_file
        with open(target_file, "r") as f:
            self.original_code = f.read()
        self.ast_tree = ast.parse(self.original_code)
    
    def modify_network(self, modification: Dict, save_callback=None) -> bool:
        try:
            if modification["method"] == "add_layer":
                self._add_layer(modification["params"])
            elif modification["method"] == "change_hyperparam":
                self._change_hyperparam(modification["params"])
            elif modification["method"] == "add_module":
                self._add_module(modification["params"])
            
            if not self.validate_code():
                return False
            
            new_code = ast.unparse(self.ast_tree)
            with open(self.target_file, "w") as f:
                f.write(new_code)
            
            if save_callback:
                save_callback()
            
            logger.info("Code modified, restarting system...")
            os.execv(sys.executable, [sys.executable] + sys.argv)
            return True
        except Exception as e:
            logger.error(f"Code modification failed: {e}")
            return False
    
    def _add_layer(self, params: Dict):
        for node in ast.walk(self.ast_tree):
            if (isinstance(node, ast.ClassDef) and node.name == "Organism" and
                isinstance(node.body, list)):
                for stmt in node.body:
                    if (isinstance(stmt, ast.Assign) and 
                        isinstance(stmt.targets[0], ast.Name) and 
                        stmt.targets[0].id == "processing"):
                        new_layer = ast.parse(
                            f"nn.Linear({params['in_features']}, {params['out_features']}).to(self.device)"
                        ).body[0].value
                        stmt.value.elts.append(new_layer)
                        stmt.value.elts.append(ast.Name(id="nn.ReLU", ctx=ast.Load()))
                        break
    
    def _change_hyperparam(self, params: Dict):
        for node in ast.walk(self.ast_tree):
            if (isinstance(node, ast.Assign) and 
                isinstance(node.targets[0], ast.Name) and 
                node.targets[0].id == "attention"):
                for kw in node.value.keywords:
                    if kw.arg == "num_heads":
                        kw.value = ast.Num(n=params["num_heads"])
                        break
    
    def _add_module(self, params: Dict):
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.ClassDef) and node.name == "Organism":
                new_module = ast.parse(
                    f"self.brain['{params['name']}'] = nn.Linear({params['in_features']}, {params['out_features']}).to(self.device)"
                ).body[0]
                node.body.append(new_module)
                break
    
    def validate_code(self) -> bool:
        try:
            ast.parse(ast.unparse(self.ast_tree))
            return True
        except SyntaxError as e:
            logger.error(f"Invalid code syntax: {e}")
            return False