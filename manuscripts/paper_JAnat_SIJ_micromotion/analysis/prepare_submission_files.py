"""Prepare editable caption sheets and figure files in manuscript order.

Run after compiling main.tex so figure and table references use current numbers.
This copies existing figures and does not recompute or alter research outputs.
"""
from pathlib import Path
import csv
import re
import shutil
import subprocess

PAPER = Path(__file__).resolve().parents[1]


def braced(text, start):
    depth = 1
    end = start
    while depth:
        char = text[end]
        if char in '{}':
            preceding = len(text[:end])-len(text[:end].rstrip('\\'))
            if preceding % 2 == 0:
                depth += 1 if char == '{' else -1
        end += 1
    return text[start:end-1]


def run():
    output = PAPER/'submission'
    figures = output/'figures'
    figures.mkdir(parents=True, exist_ok=True)
    refs = dict(re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]+)\}', (PAPER/'main.aux').read_text()))
    sheets = []
    mapping = []
    for name in ['methods', 'results']:
        for block in re.findall(r'\\begin\{figure\}.*?\\end\{figure\}', (PAPER/'sections'/f'{name}.tex').read_text(), re.S):
            label = re.search(r'\\label\{([^}]+)\}', block)[1]
            number = refs[label]
            caption_start = block.index(r'\caption{')+len(r'\caption{')
            caption = braced(block, caption_start)
            caption = re.sub(r'\\ref\{([^}]+)\}', lambda m: refs[m[1]], caption)
            sheets.append(r'\section*{Figure '+number+'}\n'+caption+'\n')
            source = PAPER/re.search(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', block)[1]
            destination = figures/f'Figure_{int(number):02d}.pdf'
            shutil.copy2(source, destination)
            mapping.append(dict(figure=number, label=label,
                                source=str(source.relative_to(PAPER)),
                                submission_file=str(destination.relative_to(PAPER))))
    latex = output/'Figure_captions.tex'
    latex.write_text('\n'.join(sheets))
    subprocess.run(['pandoc', str(latex), '-f', 'latex', '-t', 'docx', '--standalone',
                    '-o', str(output/'Figure_captions.docx')], check=True)
    subprocess.run(['pandoc', str(latex), '-f', 'latex', '-t', 'gfm',
                    '-o', str(output/'Figure_captions.md')], check=True)
    with (output/'figure_files.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mapping[0]))
        writer.writeheader()
        writer.writerows(mapping)
    print('Prepared', len(mapping), 'numbered figures and an editable caption sheet.')


if __name__ == '__main__':
    run()
