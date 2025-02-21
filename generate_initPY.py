import ast
import os
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple, NamedTuple
from collections import defaultdict

class CodeElement(NamedTuple):
    """Structure to hold code element information."""
    name: str
    type: str  # 'function', 'class', 'import', or 'type'
    module: str
    is_typing: bool = False
    parent_class: str = None  # Name of parent class if it's a method

class InitFileGenerator:
    def __init__(self, directory: str):
        self.directory = Path(directory)
        self.exclude_files = {'__init__.py'}
        self.typing_modules = {'typing', 'collections', 'dataclasses', 'pathlib'}
        
    def get_python_files(self) -> List[Path]:
        """Get all Python files in the directory excluding __init__.py."""
        return [
            f for f in self.directory.glob('*.py') 
            if f.name not in self.exclude_files
        ]

    def is_typing_import(self, module: str, name: str) -> bool:
        """Check if an import is from typing-related modules."""
        return module in self.typing_modules

    def extract_imports(self, tree: ast.AST, file_path: Path) -> Set[CodeElement]:
        """Extract imported names from the AST."""
        imports = set()
        module_name = file_path.stem
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level == 0:  # absolute imports
                    module = node.module
                    for name in node.names:
                        if not name.name.startswith('_'):
                            is_typing = self.is_typing_import(module, name.name)
                            imports.add(CodeElement(
                                name=name.asname or name.name,
                                type='type' if is_typing else 'import',
                                module=module,
                                is_typing=is_typing
                            ))
        return imports

    def get_parent_class(self, node: ast.AST) -> str:
        """Get the name of the parent class for a node."""
        for parent in ast.walk(node):
            if isinstance(parent, ast.ClassDef):
                return parent.name
        return None

    def extract_code_elements(self, file_path: Path) -> Set[CodeElement]:
        """Extract all code elements from a Python file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            print(f"Syntax error in {file_path}: {e}")
            return set()
            
        module_name = file_path.stem
        elements = set()

        # Add standalone functions (not methods)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not node.name.startswith('_'):
                    parent_class = self.get_parent_class(node)
                    if not parent_class:  # Only add if not a method
                        elements.add(CodeElement(
                            name=node.name,
                            type='function',
                            module=module_name,
                            parent_class=None
                        ))
        
        # Add classes
        elements.update({
            CodeElement(
                name=node.name,
                type='class',
                module=module_name
            )
            for node in ast.walk(tree) 
            if isinstance(node, ast.ClassDef) and not node.name.startswith('_')
        })

        # Add imports
        elements.update(self.extract_imports(tree, file_path))
        
        return elements

    def organize_imports(self, elements: Set[CodeElement]) -> Dict[str, List[str]]:
        """Organize imports by their source modules."""
        typing_imports = defaultdict(set)
        module_imports = defaultdict(set)
        
        for elem in elements:
            if elem.is_typing:
                typing_imports[elem.module].add(elem.name)
            elif elem.type in ('function', 'class'):
                module_imports[f".{elem.module}"].add(elem.name)
        
        return {
            module: sorted(names)
            for imports_dict in (typing_imports, module_imports)
            for module, names in imports_dict.items()
        }

    def generate_init_content(self) -> str:
        """Generate content for __init__.py file."""
        all_elements = set()
        
        for file_path in self.get_python_files():
            all_elements.update(self.extract_code_elements(file_path))

        # Organize imports
        organized_imports = self.organize_imports(all_elements)

        # Separate elements by type (excluding methods)
        classes = sorted(elem.name for elem in all_elements if elem.type == 'class')
        functions = sorted(elem.name for elem in all_elements 
                         if elem.type == 'function' and not elem.parent_class)

        # Generate typing imports first
        import_lines = []
        for module, names in organized_imports.items():
            if module in self.typing_modules:
                names_str = ', '.join(sorted(names))
                import_lines.append(f"from {module} import {names_str}")
        
        # Then generate module imports
        module_imports = [
            f"from {module} import {', '.join(names)}"
            for module, names in organized_imports.items()
            if module not in self.typing_modules
        ]
        import_lines.extend(sorted(module_imports))

        # Generate __all__ (excluding typing imports and methods)
        exports = sorted([
            elem.name for elem in all_elements 
            if not elem.is_typing and elem.type in ('class', 'function') 
            and not elem.parent_class
        ])
        all_str = '[\n    ' + ',\n    '.join(f"'{name}'" for name in exports) + '\n]'
        
        # Generate docstring
        docstring = [
            '"""',
            f'Automatically generated __init__.py for {self.directory.name}',
            '',
            'This module exports the following:',
            '- Classes: ' + ', '.join(classes) if classes else '- Classes: None',
            '- Functions: ' + ', '.join(functions) if functions else '- Functions: None',
            '"""',
        ]
        
        return '\n'.join([
            *docstring,
            '',
            *import_lines,
            '',
            '__all__ = ' + all_str,
            ''
        ])

    def write_init_file(self) -> None:
        """Write the generated content to __init__.py."""
        init_path = self.directory / '__init__.py'
        content = self.generate_init_content()
        
        with open(init_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Generated {init_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Generate __init__.py file for a Python package directory'
    )
    parser.add_argument(
        'directory',
        help='Directory to generate __init__.py for'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print the content without writing to file'
    )
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        return

    generator = InitFileGenerator(args.directory)
    if args.dry_run:
        print(generator.generate_init_content())
    else:
        generator.write_init_file()

if __name__ == '__main__':
    main()