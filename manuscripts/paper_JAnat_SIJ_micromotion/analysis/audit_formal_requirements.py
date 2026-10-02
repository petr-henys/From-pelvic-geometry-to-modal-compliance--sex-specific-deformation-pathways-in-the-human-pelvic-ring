"""Local formal audit and reproducible prose word counts for the JTB manuscript.

This checks document contents, not live journal rules or submission-portal forms.
Pandoc parses LaTeX; prose counts omit headings, citations, floats and displayed
math, with each inline mathematical expression counted as one unit.
"""
from pathlib import Path
import json
import re
import subprocess

PAPER = Path(__file__).resolve().parents[1]
OUT = PAPER / 'review/formal_requirements_2026-10-02'
OUT.mkdir(parents=True, exist_ok=True)


def parsed(tex):
    result = subprocess.run(['pandoc', '-f', 'latex', '-t', 'json'], input=tex,
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def visible(node):
    if isinstance(node, list):
        return ''.join(visible(item) for item in node)
    if not isinstance(node, dict):
        return ''
    kind = node.get('t')
    if kind == 'Str':
        return node['c']
    if kind in ['Space', 'SoftBreak', 'LineBreak']:
        return ' '
    if kind == 'Math':
        return ' MATH ' if node['c'][0] == 'InlineMath' else ''
    if kind in ['Cite', 'Image', 'RawInline', 'RawBlock', 'Note']:
        return ''
    if kind == 'Link':
        return visible(node['c'][1])
    return visible(node.get('c', []))


def prose(tex):
    tex = re.sub(r'\\begin\{(figure|table)\}(?:\[[^]]*\])?.*?\\end\{\1\}', '', tex, flags=re.S)
    text = '\n'.join(visible(b.get('c', [])) for b in parsed(tex)['blocks']
                     if b['t'] in ['Para', 'Plain'])
    return text


def count(text):
    return sum(bool(re.search(r'\w', token)) for token in text.split())


def run():
    sections = ['abstract', 'introduction', 'methods', 'results', 'discussion', 'conclusion', 'appendix']
    counts = {name: count(prose((PAPER/'sections'/f'{name}.tex').read_text())) for name in sections}
    counts['main_body_excluding_abstract_appendix_captions_tables_references'] = sum(counts[s] for s in sections[1:6])
    counts['main_body_plus_abstract'] = counts['main_body_excluding_abstract_appendix_captions_tables_references']+counts['abstract']
    main = (PAPER/'main.tex').read_text()
    keys = re.search(r'\\textbf\{Keywords:\}\s*([^\n]+)', main)[1].split(';')
    counts['keywords'] = len(keys)
    titles = re.findall(r'\\title\{([^}]+)\}', main)
    counts['title'] = count(titles[0])
    floats = []
    for name in ['methods', 'results']:
        tex = (PAPER/'sections'/f'{name}.tex').read_text()
        for match in re.finditer(r'\\begin\{(figure|table)\}.*?\\end\{\1\}', tex, re.S):
            kind, block = match[1], match[0]
            label = re.search(r'\\label\{([^}]+)\}', block)
            assert label and r'\caption{' in block
            floats.append(dict(kind=kind, label=label[1]))
            for asset in re.findall(r'\\(?:includegraphics(?:\[[^]]*\])?|StdTableInput)\{([^}]+)\}', block):
                assert (PAPER/asset).exists(), asset
    counts['main_figures'] = sum(f['kind'] == 'figure' for f in floats)
    counts['main_tables'] = sum(f['kind'] == 'table' for f in floats)
    aux = (PAPER/'main.aux').read_text()
    cited = {key for group in re.findall(r'\\citation\{([^}]+)\}', aux) for key in group.split(',')}
    bibliography = set(re.findall(r'\\bibitem(?:\[[^]]*\])?\{([^}]+)\}', (PAPER/'main.bbl').read_text()))
    assert cited == bibliography, (cited-bibliography, bibliography-cited)
    counts['references'] = len(bibliography)
    present = {
        'funding': r'\section*{Funding}' in main,
        'funder_role': 'The funder had no role' in main,
        'credit_contributions': r'\section*{Author contributions}' in main,
        'competing_interest': r'\section*{Declaration of competing interest}' in main,
        'data_availability': r'\section*{Data Availability Statement}' in main,
        'AI_declaration_immediately_before_references': bool(re.search(r'\\section\*\{Declaration of generative AI.*?\\bibliographystyle', main, re.S)),
        'AI_methods': 'AI assistance and reproducible postprocessing' in (PAPER/'sections/methods.tex').read_text(),
        'ethics_approval_and_waiver': all(word in (PAPER/'sections/methods.tex').read_text() for word in ['202411IO3P', 'waiver of informed consent']),
        'author_email': 'petr.henys@tul.cz' in main,
        'author_postal_address': '461~17 Liberec' in main,
        'no_empty_acknowledgements': r'\section*{Acknowledgements}' not in main,
        'separate_competing_interest_docx': (PAPER/'submission/Declaration_of_competing_interest.docx').exists(),
        'separate_highlights_docx': (PAPER/'submission/Highlights.docx').exists(),
        'separate_figure_captions_docx': (PAPER/'submission/Figure_captions.docx').exists(),
    }
    assert all(present.values()), present
    highlights = [line[2:] for line in (PAPER/'submission/Highlights.md').read_text().splitlines() if line.startswith('- ')]
    lengths = [len(line) for line in highlights]
    assert 3 <= len(highlights) <= 5 and max(lengths) <= 85
    # Conservative local preparation targets, not an assertion of live JTB limits.
    assert counts['abstract'] <= 250 and counts['keywords'] <= 5
    font_check = {}
    assets = [PAPER/re.search(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',
                     match[0])[1] for name in ['methods', 'results']
                     for match in re.finditer(r'\\begin\{figure\}.*?\\end\{figure\}', (PAPER/'sections'/f'{name}.tex').read_text(), re.S)]
    supp = PAPER/'supplementary/supplement.tex'
    assets.extend((supp.parent/path).resolve() for path in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', supp.read_text()))
    for f in [PAPER/'main.pdf', PAPER/'supplementary/supplement.pdf', *assets]:
        lines = subprocess.run(['pdffonts', str(f)], capture_output=True, text=True, check=True).stdout.splitlines()[2:]
        assert all('Type 3' not in line for line in lines), f
        for line in lines:
            # Last fields are emb/sub/uni/object/ID; object and ID occupy two tokens.
            assert line.split()[-5] == 'yes', (f, line)
        font_check[str(f.relative_to(PAPER))] = dict(fonts=len(lines), all_embedded=True, type3=False)
    result = dict(date='2026-10-02', word_counts=counts,
                  counting_method='Pandoc LaTeX AST; prose only; headings, captions, tables, citations and displayed equations excluded; each inline expression counts as one unit; hyphenated/range tokens count as one',
                  declarations=present, highlights_characters_including_spaces=lengths,
                  bibliography_citation_pairs_match=True, fonts=font_check,
                  journal_guide_verification='Current JTB Guide for Authors could not be retrieved (HTTP 403); current Elsevier AI/Highlights/submission/data policies verified separately; numerical preparation targets are conservative and not fully certified live journal rules')
    (OUT/'formal_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != 'fonts'}, indent=2))


if __name__ == '__main__':
    run()
