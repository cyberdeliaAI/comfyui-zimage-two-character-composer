"""Build standalone installation and development archives."""
from pathlib import Path
import re
import zipfile

root = Path(__file__).resolve().parents[1]
version = re.search(r'^version = "([^"]+)"', (root/'pyproject.toml').read_text(), re.MULTILINE).group(1)
files = [root/name for name in ('__init__.py', 'composer.py', 'regions.py', 'prompting.py',
                               'negpip_bridge.py', 'pyproject.toml', 'README.md', 'CHANGELOG.md', 'LICENSE')]
for directory in ('web', 'assets', 'example_workflows'):
    files.extend(p for p in (root/directory).rglob('*') if p.is_file())
source_files = files + [root/'.gitignore', root/'.comfyignore']
for directory in ('tests', 'tools', '.github'):
    source_files.extend(p for p in (root/directory).rglob('*') if p.is_file()
                        and '__pycache__' not in p.parts and p.suffix != '.pyc')
(root/'dist').mkdir(exist_ok=True)
for suffix, selected in (('', files), ('-source', source_files)):
    path = root/'dist'/f'comfyui-zimage-two-character-composer-{version}{suffix}.zip'
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(set(selected)):
            archive.write(item, Path('comfyui-zimage-two-character-composer')/item.relative_to(root))
    print(path)
