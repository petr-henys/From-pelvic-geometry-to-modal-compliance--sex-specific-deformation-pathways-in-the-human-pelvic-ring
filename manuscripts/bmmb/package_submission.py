"""Package the manuscript and cover letter and their local source dependencies."""
from pathlib import Path
import re
from zipfile import ZipFile, ZIP_DEFLATED

root = Path(__file__).resolve().parent
files = set()
pattern = re.compile(r'\\(?:input|includegraphics|StdTableInput|DetailTableInput)(?:\[[^\]]*\])?\{([^}]+)\}')

def add(name):
    if name in files:
        return
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(path)
    files.add(name)
    if path.suffix == '.tex':
        for dependency in pattern.findall(path.read_text()):
            if '#' not in dependency:
                add(dependency)

for stem in ('main_bmmb', 'cover_letter'):
    add(stem + '.tex')
    add(stem + '.pdf')
for name in ('main_bmmb.bbl', 'sn-jnl.cls', 'sn-basic.bst',
             'references.bib', 'cover_letter.md'):
    add(name)
with ZipFile(root / 'bmmb_submission.zip', 'w', ZIP_DEFLATED) as archive:
    for name in sorted(files):
        archive.write(root / name, name)
print(f'Packaged {len(files)} files in bmmb_submission.zip')
