# Helper script to generate both complete styled HTML and clean WYSIWYG snippet
import markdown
from pathlib import Path

ROOT = Path(__file__).resolve().parent

with open(ROOT / "ZENODO_CODE_STRUCTURE.md", "r", encoding="utf-8") as f:
    md_content = f.read()

# Convert markdown to clean HTML
html_body = markdown.markdown(md_content, extensions=["tables", "fenced_code"])

# 1. Full styled document (for opening in browser and copying rich-text directly into WYSIWYG)
full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Zenodo Description - Pelvic Ring Biomechanics</title>
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.6;
    color: #24292e;
    max-width: 960px;
    margin: 40px auto;
    padding: 0 20px;
}}
h1, h2, h3, h4 {{ color: #111827; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }}
h1 {{ font-size: 2em; }}
h2 {{ font-size: 1.5em; margin-top: 28px; }}
h3 {{ font-size: 1.25em; margin-top: 22px; border-bottom: none; }}
h4 {{ font-size: 1.05em; margin-top: 18px; border-bottom: none; }}
table {{
    border-collapse: collapse;
    width: 100%;
    margin: 20px 0;
}}
th, td {{
    border: 1px solid #d0d7de;
    padding: 10px 14px;
    text-align: left;
}}
th {{
    background-color: #f6f8fa;
    font-weight: 600;
}}
tr:nth-child(even) {{ background-color: #fcfcfc; }}
code {{
    background-color: #f6f8fa;
    border: 1px solid #e1e4e8;
    border-radius: 4px;
    padding: 0.2em 0.4em;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 88%;
}}
pre {{
    background-color: #f6f8fa;
    border: 1px solid #e1e4e8;
    border-radius: 6px;
    padding: 14px 18px;
    overflow-x: auto;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 85%;
    line-height: 1.45;
}}
pre code {{
    background-color: transparent;
    border: none;
    padding: 0;
}}
ul, ol {{ padding-left: 2em; }}
li {{ margin: 0.3em 0; }}
a {{ color: #0969da; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
hr {{ height: 1px; background-color: #e1e4e8; border: none; margin: 28px 0; }}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

with open(ROOT / "ZENODO_DESCRIPTION.html", "w", encoding="utf-8") as f:
    f.write(full_html)

# 2. Raw HTML snippet (for pasting directly into Zenodo's "<>" / Source code editor)
with open(ROOT / "ZENODO_DESCRIPTION_SNIPPET.html", "w", encoding="utf-8") as f:
    f.write(html_body)

print("Successfully generated:")
print(" - ZENODO_DESCRIPTION.html (Open in browser -> Ctrl+A -> Ctrl+C -> Paste in Zenodo WYSIWYG)")
print(" - ZENODO_DESCRIPTION_SNIPPET.html (Paste directly into Zenodo Source / <> mode)")

