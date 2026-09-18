from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MANUSCRIPT_PATH = SCRIPT_DIR / "main_plos.tex"
BBL_PATH = SCRIPT_DIR / "main_plos.bbl"
OUTPUT_PATH = SCRIPT_DIR / "main_plos.docx"

INPUT_PATTERN = re.compile(r"\\(?:StdTableInput|SuppTableInput|input)\{([^}]+)\}")
INCLUDEGRAPHICS_PATTERN = re.compile(
    r"\\includegraphics(?:\[(?P<options>[^\]]*)\])?\{(?P<path>[^}]+)\}"
)
CITATION_PATTERN = re.compile(r"\\cite[a-zA-Z*]*\{([^}]+)\}")
INLINE_RM_PATTERN = re.compile(r"([_^])\{\\rm\s+([^{}]+)\}")
LABEL_PATTERN = re.compile(r"\\label\{([^}]+)\}")
REF_PATTERN = re.compile(r"\\ref\{([^}]+)\}")
EQREF_PATTERN = re.compile(r"\\eqref\{([^}]+)\}")
TOKEN_PATTERN = re.compile(
    r"\\appendix\b"
    r"|\\section(\*?)\{" 
    r"|\\subsection(\*?)\{" 
    r"|\\begin\{(figure|table|equation)\}"
    r"|\\end\{(figure|table|equation)\}"
    r"|\\label\{([^}]+)\}"
)
CAPTION_TOKEN_PATTERN = re.compile(
    r"(?P<begin>\\begin\{(?P<begin_env>figure|table)\})"
    r"|(?P<end>\\end\{(?P<end_env>figure|table)\})"
    r"|(?P<caption>\\caption(?:\[[^\]]*\])?\{)"
)
BBL_ITEM_PATTERN = re.compile(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}")


def resolve_input_path(base_dir: Path, raw_path: str) -> Path:
    candidate = (base_dir / raw_path).resolve()
    if candidate.exists():
        return candidate
    if candidate.suffix:
        raise FileNotFoundError(f"Included file not found: {raw_path}")

    tex_candidate = candidate.with_suffix(".tex")
    if tex_candidate.exists():
        return tex_candidate

    raise FileNotFoundError(f"Included file not found: {raw_path}")


def inline_inputs(text: str, base_dir: Path) -> str:
    def replace(match: re.Match[str]) -> str:
        raw_path = match.group(1).strip()
        if "#" in raw_path:
            return match.group(0)

        include_path = resolve_input_path(base_dir, raw_path)
        included_text = include_path.read_text(encoding="utf-8")
        expanded_text = inline_inputs(included_text, include_path.parent)
        return f"\n{expanded_text}\n"

    return INPUT_PATTERN.sub(replace, text)


def build_citation_map(bbl_text: str) -> dict[str, int]:
    citation_map: dict[str, int] = {}
    for index, match in enumerate(BBL_ITEM_PATTERN.finditer(bbl_text), start=1):
        citation_map[match.group(1)] = index
    return citation_map


def compress_numbers(numbers: list[int]) -> str:
    ordered_numbers = sorted(dict.fromkeys(numbers))
    if not ordered_numbers:
        return "[]"

    ranges: list[str] = []
    range_start = ordered_numbers[0]
    range_end = ordered_numbers[0]
    for number in ordered_numbers[1:]:
        if number == range_end + 1:
            range_end = number
            continue

        if range_start == range_end:
            ranges.append(str(range_start))
        else:
            ranges.append(f"{range_start}-{range_end}")
        range_start = number
        range_end = number

    if range_start == range_end:
        ranges.append(str(range_start))
    else:
        ranges.append(f"{range_start}-{range_end}")
    return f"[{','.join(ranges)}]"


def replace_citations(text: str, citation_map: dict[str, int]) -> str:
    def replace(match: re.Match[str]) -> str:
        keys = [key.strip() for key in match.group(1).split(",") if key.strip()]
        missing_keys = [key for key in keys if key not in citation_map]
        if missing_keys:
            missing_text = ", ".join(missing_keys)
            raise ValueError(f"Missing citation keys in main_plos.bbl: {missing_text}")

        citation_numbers = [citation_map[key] for key in keys]
        return compress_numbers(citation_numbers)

    return CITATION_PATTERN.sub(replace, text)


def normalize_math_markup(text: str) -> str:
    text = INLINE_RM_PATTERN.sub(r"\1{\\mathrm{\2}}", text)
    return text.replace(r"\geq", r"\ge")


def format_section_index(index: int, appendix_mode: bool) -> str:
    if appendix_mode:
        return chr(ord("A") + index - 1)
    return str(index)


def build_reference_map(text: str) -> dict[str, str]:
    reference_map: dict[str, str] = {}
    appendix_mode = False
    section_index = 0
    subsection_index = 0
    figure_index = 0
    table_index = 0
    equation_index = 0
    current_environment: tuple[str, str] | None = None
    pending_section_number: str | None = None

    for match in TOKEN_PATTERN.finditer(text):
        token = match.group(0)
        if token == r"\appendix":
            appendix_mode = True
            section_index = 0
            subsection_index = 0
            pending_section_number = None
            continue

        section_star = match.group(1)
        subsection_star = match.group(2)
        begin_environment = match.group(3)
        end_environment = match.group(4)
        label_name = match.group(5)

        if section_star is not None:
            if section_star == "*":
                pending_section_number = None
                continue

            section_index += 1
            subsection_index = 0
            pending_section_number = format_section_index(section_index, appendix_mode)
            continue

        if subsection_star is not None:
            if subsection_star == "*":
                pending_section_number = None
                continue

            subsection_index += 1
            section_number = format_section_index(section_index, appendix_mode)
            pending_section_number = f"{section_number}.{subsection_index}"
            continue

        if begin_environment is not None:
            if begin_environment == "figure":
                figure_index += 1
                current_environment = ("figure", str(figure_index))
            elif begin_environment == "table":
                table_index += 1
                current_environment = ("table", str(table_index))
            elif begin_environment == "equation":
                equation_index += 1
                current_environment = ("equation", str(equation_index))
            continue

        if end_environment is not None:
            current_environment = None
            continue

        if label_name is None:
            continue

        if current_environment is not None:
            reference_map[label_name] = current_environment[1]
            continue

        if pending_section_number is not None:
            reference_map[label_name] = pending_section_number
            pending_section_number = None

    return reference_map


def replace_references(text: str, reference_map: dict[str, str]) -> str:
    def replace_eqref(match: re.Match[str]) -> str:
        label = match.group(1)
        if label not in reference_map:
            raise ValueError(f"Missing equation reference target: {label}")
        return f"({reference_map[label]})"

    def replace_ref(match: re.Match[str]) -> str:
        label = match.group(1)
        if label not in reference_map:
            raise ValueError(f"Missing reference target: {label}")
        return reference_map[label]

    text = EQREF_PATTERN.sub(replace_eqref, text)
    return REF_PATTERN.sub(replace_ref, text)


def replace_bibliography(text: str, bibliography_text: str) -> str:
    bibliography_block = "\\section*{References}\n" + bibliography_text
    return re.sub(
        r"\\bibliographystyle\{[^}]+\}\s*\\bibliography\{[^}]+\}",
        lambda _: bibliography_block,
        text,
        count=1,
        flags=re.S,
    )


def add_caption_prefixes(text: str) -> str:
    figure_index = 0
    table_index = 0
    current_environment: str | None = None
    current_index: int | None = None
    caption_prefixed = False
    rebuilt_text: list[str] = []
    last_end = 0

    for match in CAPTION_TOKEN_PATTERN.finditer(text):
        rebuilt_text.append(text[last_end:match.start()])

        begin_environment = match.group("begin_env")
        end_environment = match.group("end_env")
        caption_token = match.group("caption")

        if begin_environment is not None:
            current_environment = begin_environment
            caption_prefixed = False
            if begin_environment == "figure":
                figure_index += 1
                current_index = figure_index
            else:
                table_index += 1
                current_index = table_index
            rebuilt_text.append(match.group("begin"))
        elif end_environment is not None:
            current_environment = None
            current_index = None
            caption_prefixed = False
            rebuilt_text.append(match.group("end"))
        elif (
            caption_token is not None
            and current_environment in {"figure", "table"}
            and current_index is not None
            and not caption_prefixed
        ):
            prefix = "Figure" if current_environment == "figure" else "Table"
            rebuilt_text.append(f"{caption_token}{prefix} {current_index}. ")
            caption_prefixed = True
        else:
            rebuilt_text.append(match.group(0))

        last_end = match.end()

    rebuilt_text.append(text[last_end:])
    return "".join(rebuilt_text)


def convert_pdf_to_png(source_path: Path, media_dir: Path) -> Path:
    file_hash = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:12]
    output_base = media_dir / f"{source_path.stem}_{file_hash}"
    output_path = output_base.with_suffix(".png")
    if output_path.exists():
        return output_path

    subprocess.run(
        ["pdftoppm", "-png", "-singlefile", str(source_path), str(output_base)],
        check=True,
    )
    return output_path


def normalize_graphics_paths(text: str, manuscript_dir: Path, media_dir: Path) -> str:
    def replace(match: re.Match[str]) -> str:
        options = match.group("options")
        raw_path = match.group("path").strip()
        resolved_path = resolve_input_path(manuscript_dir, raw_path)
        if resolved_path.suffix.lower() == ".pdf":
            png_path = convert_pdf_to_png(resolved_path, media_dir)
            resolved_path = png_path

        option_text = f"[{options}]" if options else ""
        return f"\\includegraphics{option_text}{{{resolved_path.as_posix()}}}"

    return INCLUDEGRAPHICS_PATTERN.sub(replace, text)


def remove_labels(text: str) -> str:
    return LABEL_PATTERN.sub("", text)


def ensure_external_tools() -> None:
    required_tools = ["pandoc"]
    optional_pdf_tool_needed = MANUSCRIPT_PATH.read_text(encoding="utf-8").count(".pdf") > 0
    if optional_pdf_tool_needed:
        required_tools.append("pdftoppm")

    missing_tools = [tool for tool in required_tools if shutil.which(tool) is None]
    if missing_tools:
        missing_text = ", ".join(missing_tools)
        raise RuntimeError(f"Missing required external tools: {missing_text}")


def build_export_source(media_dir: Path) -> str:
    manuscript_text = MANUSCRIPT_PATH.read_text(encoding="utf-8")
    bbl_text = BBL_PATH.read_text(encoding="utf-8")
    reference_map = build_reference_map(manuscript_text)
    citation_map = build_citation_map(bbl_text)

    manuscript_text = replace_bibliography(manuscript_text, bbl_text)
    manuscript_text = inline_inputs(manuscript_text, SCRIPT_DIR)
    manuscript_text = add_caption_prefixes(manuscript_text)
    manuscript_text = replace_citations(manuscript_text, citation_map)
    manuscript_text = replace_references(manuscript_text, reference_map)
    manuscript_text = remove_labels(manuscript_text)
    manuscript_text = normalize_math_markup(manuscript_text)
    manuscript_text = normalize_graphics_paths(manuscript_text, SCRIPT_DIR, media_dir)
    return manuscript_text


def export_to_docx() -> Path:
    ensure_external_tools()
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        media_dir = tmp_dir / "media"
        media_dir.mkdir(parents=True, exist_ok=True)

        export_text = build_export_source(media_dir)
        export_source_path = tmp_dir / "main_plos_export.tex"
        export_source_path.write_text(export_text, encoding="utf-8")

        subprocess.run(
            [
                "pandoc",
                str(export_source_path),
                "--from=latex",
                "--to=docx",
                f"--output={OUTPUT_PATH}",
            ],
            check=True,
            cwd=SCRIPT_DIR,
        )
    return OUTPUT_PATH


if __name__ == "__main__":
    output_path = export_to_docx()
    print(output_path)