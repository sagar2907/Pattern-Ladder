"""Render a Markdown document to PDF, and verify the result visually.

Why this is hand-rolled rather than a call to pandoc or weasyprint: both need a
system toolchain (LaTeX, or GTK/Cairo) that is not present on a plain Windows
or CI machine, which would make "render the report" a step that works on one
person's laptop. reportlab is pure Python and already a dependency.

The verification step is the point of the file. A missing glyph in a PDF does
not raise -- it renders as a black box or as nothing at all, and the document
looks fine to the program that produced it. So two checks run:

  1. Before rendering, every character in the source is checked against the
     chosen font's character map, and anything unsupported is substituted for
     an ASCII equivalent (and reported), rather than being emitted and hoped
     for.
  2. After rendering, each page is rasterised to an image and inspected for
     the signature of a glyph failure: a page that is almost entirely blank,
     or one carrying an implausible amount of solid black.

Run: python scripts/render_pdf.py [source.md] [-o out.pdf] [--images DIR]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)

DOCS = Path(__file__).resolve().parent.parent / "docs"

# Font families are tried in order. Each entry is (family, regular, bold,
# italic, bold-italic). Vera ships inside reportlab, so the last entry always
# resolves and the script never depends on a system font being installed.
FONT_CANDIDATES = [
    (
        "DejaVu",
        "DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf",
        "DejaVuSans-Oblique.ttf",
        "DejaVuSans-BoldOblique.ttf",
    ),
    ("Vera", "Vera.ttf", "VeraBd.ttf", "VeraIt.ttf", "VeraBI.ttf"),
]

MONO_CANDIDATES = [
    ("DejaVuMono", "DejaVuSansMono.ttf", "DejaVuSansMono-Bold.ttf"),
    ("VeraMono", "VeraMono.ttf", "VeraMoBd.ttf"),
]

# Characters worth using for readability, each with an ASCII fallback for when
# the resolved font cannot draw them. Substituting deliberately is far better
# than shipping a document full of black rectangles.
SUBSTITUTIONS = {
    "\u2014": "--", "\u2013": "-", "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u2192": "->",
    "\u2190": "<-", "\u00b7": "-", "\u2022": "*", "\u2265": ">=",
    "\u2264": "<=", "\u00d7": "x", "\u2248": "~", "\u2260": "!=",
    "\u00a0": " ", "\u2713": "yes", "\u2717": "no", "\u00b1": "+/-",
}


def _font_search_paths() -> list[Path]:
    import reportlab

    paths = [Path(reportlab.__file__).parent / "fonts"]
    paths += [
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/TTF"),
        Path("/Library/Fonts"),
        Path("C:/Windows/Fonts"),
    ]
    try:
        import matplotlib

        paths.insert(0, Path(matplotlib.get_data_path()) / "fonts" / "ttf")
    except ImportError:
        pass
    return [p for p in paths if p.is_dir()]


def _locate(filename: str, search: list[Path]) -> Path | None:
    for directory in search:
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return None


def register_fonts() -> tuple[str, str, set[int]]:
    """Register the best available font family. Returns (body, mono, charset)."""
    search = _font_search_paths()

    body = None
    for family, regular, bold, italic, bold_italic in FONT_CANDIDATES:
        found = {
            style: _locate(name, search)
            for style, name in (
                ("", regular), ("-Bold", bold), ("-Italic", italic), ("-BoldItalic", bold_italic)
            )
        }
        if all(found.values()):
            for style, path in found.items():
                pdfmetrics.registerFont(TTFont(f"{family}{style}", str(path)))
            pdfmetrics.registerFontFamily(
                family,
                normal=family,
                bold=f"{family}-Bold",
                italic=f"{family}-Italic",
                boldItalic=f"{family}-BoldItalic",
            )
            body = family
            break
    if body is None:
        raise RuntimeError("no usable body font found")

    mono = "Courier"
    for family, regular, bold in MONO_CANDIDATES:
        regular_path, bold_path = _locate(regular, search), _locate(bold, search)
        if regular_path and bold_path:
            pdfmetrics.registerFont(TTFont(family, str(regular_path)))
            pdfmetrics.registerFont(TTFont(f"{family}-Bold", str(bold_path)))
            pdfmetrics.registerFontFamily(family, normal=family, bold=f"{family}-Bold")
            mono = family
            break

    # The character map of the registered body font, used to catch glyphs that
    # would otherwise silently fail to draw.
    face = pdfmetrics.getFont(body).face
    charset = set(getattr(face, "charToGlyph", {}).keys())
    return body, mono, charset


def sanitise(text: str, charset: set[int]) -> tuple[str, set[str]]:
    """Replace characters the font cannot draw. Returns (text, replaced)."""
    if not charset:
        return text, set()
    replaced: set[str] = set()
    out = []
    for char in text:
        if ord(char) in charset or char in "\n\t":
            out.append(char)
            continue
        replacement = SUBSTITUTIONS.get(char)
        if replacement is None:
            # Unknown and undrawable: drop it rather than emit a black box,
            # and report it so the source can be fixed.
            replacement = ""
        replaced.add(char)
        out.append(replacement)
    return "".join(out), replaced


# --- Markdown ----------------------------------------------------------------

_INLINE_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def inline(text: str, mono: str) -> str:
    """Convert inline Markdown to reportlab's mini-HTML.

    Escaping happens first: a stray ampersand or angle bracket in the source
    would otherwise be parsed as markup and abort the render.
    """
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = _LINK.sub(r'<link href="\2" color="#0b5ed7">\1</link>', text)
    text = _BOLD.sub(r"<b>\1</b>", text)
    text = _ITALIC.sub(r"<i>\1</i>", text)
    return _INLINE_CODE.sub(
        rf'<font face="{mono}" size="9" backColor="#f0f0f2">\1</font>', text
    )


class MarkdownRenderer:
    def __init__(self, body: str, mono: str) -> None:
        self.body = body
        self.mono = mono
        base = getSampleStyleSheet()["BodyText"]
        self.styles = {
            "body": ParagraphStyle(
                "body", parent=base, fontName=body, fontSize=9.7, leading=14.6,
                spaceAfter=7, alignment=TA_LEFT, textColor=colors.HexColor("#16181d"),
            ),
            "h1": ParagraphStyle(
                "h1", parent=base, fontName=f"{body}-Bold", fontSize=21, leading=26,
                spaceBefore=6, spaceAfter=12, textColor=colors.HexColor("#0d1117"),
            ),
            "h2": ParagraphStyle(
                "h2", parent=base, fontName=f"{body}-Bold", fontSize=15.5, leading=20,
                spaceBefore=17, spaceAfter=8, textColor=colors.HexColor("#0d1117"),
            ),
            "h3": ParagraphStyle(
                "h3", parent=base, fontName=f"{body}-Bold", fontSize=12, leading=16,
                spaceBefore=13, spaceAfter=6, textColor=colors.HexColor("#22262d"),
            ),
            "h4": ParagraphStyle(
                "h4", parent=base, fontName=f"{body}-Bold", fontSize=10.3, leading=14,
                spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#3a4048"),
            ),
            "bullet": ParagraphStyle(
                "bullet", parent=base, fontName=body, fontSize=9.7, leading=14.2,
                leftIndent=13, bulletIndent=3, spaceAfter=3.5,
                textColor=colors.HexColor("#16181d"),
            ),
            "quote": ParagraphStyle(
                "quote", parent=base, fontName=f"{body}-Italic", fontSize=9.5,
                leading=14, leftIndent=12, spaceAfter=8,
                textColor=colors.HexColor("#4a5058"),
            ),
        }
        self.code_style = ParagraphStyle(
            "code", fontName=mono, fontSize=8.1, leading=10.6,
            textColor=colors.HexColor("#1a1d23"),
        )

    def convert(self, markdown: str) -> list:
        story: list = []
        lines = markdown.split("\n")
        index = 0
        while index < len(lines):
            line = lines[index]
            stripped = line.strip()

            if stripped.startswith("```"):
                index, block = self._code_block(lines, index)
                story.append(block)
                continue

            if stripped.startswith("|") and self._is_table(lines, index):
                index, block = self._table(lines, index)
                story.append(block)
                continue

            if not stripped:
                index += 1
                continue

            if stripped in ("---", "***", "___"):
                # A rule directly before a part heading is dropped. Level-1
                # headings force a page break, so the rule would be the last
                # thing on the outgoing page -- and when the preceding part
                # happens to end near the bottom, the rule spills onto a page of
                # its own and the break then pushes the heading to the page
                # after that. The result is a page containing one horizontal
                # line and a footer, which is what the blank-page check flagged.
                # The page break already separates the parts; the rule is
                # redundant there rather than merely unlucky.
                if _next_is_part_heading(lines, index + 1):
                    index += 1
                    continue
                story.append(Spacer(1, 3))
                story.append(
                    HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#d6d9de"))
                )
                story.append(Spacer(1, 7))
                index += 1
                continue

            if stripped.startswith("#"):
                level = len(stripped) - len(stripped.lstrip("#"))
                text = stripped[level:].strip()
                if level == 1 and story:
                    story.append(PageBreak())
                style = self.styles.get(f"h{min(level, 4)}", self.styles["h4"])
                story.append(Paragraph(inline(text, self.mono), style))
                index += 1
                continue

            if stripped.startswith(">"):
                # Consecutive quote lines are one block quote, as in Markdown.
                # Rendering each line as its own paragraph leaves gaps that
                # read as unrelated fragments.
                quote: list[str] = []
                while index < len(lines) and lines[index].strip().startswith(">"):
                    quote.append(lines[index].strip().lstrip(">").strip())
                    index += 1
                story.append(
                    Paragraph(inline(" ".join(q for q in quote if q), self.mono),
                              self.styles["quote"])
                )
                continue

            bullet = re.match(r"^([-*+])\s+(.*)$", stripped)
            if bullet:
                story.append(
                    Paragraph(
                        inline(bullet.group(2), self.mono),
                        self.styles["bullet"],
                        bulletText="\u2022" if "\u2022" else "-",
                    )
                )
                index += 1
                continue

            numbered = re.match(r"^(\d+)\.\s+(.*)$", stripped)
            if numbered:
                story.append(
                    Paragraph(
                        inline(numbered.group(2), self.mono),
                        self.styles["bullet"],
                        bulletText=f"{numbered.group(1)}.",
                    )
                )
                index += 1
                continue

            # Consecutive non-blank lines form one paragraph, as in Markdown.
            buffer = [stripped]
            index += 1
            while index < len(lines):
                nxt = lines[index].strip()
                if (
                    not nxt
                    or nxt.startswith(("#", "```", "|", ">"))
                    or re.match(r"^([-*+]|\d+\.)\s+", nxt)
                    or nxt in ("---", "***", "___")
                ):
                    break
                buffer.append(nxt)
                index += 1
            story.append(Paragraph(inline(" ".join(buffer), self.mono), self.styles["body"]))
        return story

    def _code_block(self, lines: list[str], index: int):
        language = lines[index].strip()[3:].strip()  # noqa: F841 - kept for clarity
        index += 1
        body: list[str] = []
        while index < len(lines) and not lines[index].strip().startswith("```"):
            body.append(lines[index])
            index += 1
        index += 1

        # Long code lines are truncated rather than allowed to overflow the
        # frame, where reportlab would silently clip them mid-glyph.
        width = 96
        rendered = "\n".join(
            line if len(line) <= width else line[: width - 1] + "\u203a" for line in body
        )
        table = Table(
            [[Preformatted(rendered, self.code_style)]],
            colWidths=[165 * mm],
        )
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f6f7f9")),
                    ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#dfe2e7")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ]
            )
        )
        return index, KeepTogether([Spacer(1, 2), table, Spacer(1, 9)])

    @staticmethod
    def _is_table(lines: list[str], index: int) -> bool:
        return (
            index + 1 < len(lines)
            and set(lines[index + 1].strip()) <= set("|-: ")
            and "-" in lines[index + 1]
        )

    def _table(self, lines: list[str], index: int):
        def cells(row: str) -> list[str]:
            return [c.strip() for c in row.strip().strip("|").split("|")]

        header = cells(lines[index])
        index += 2
        rows = []
        while index < len(lines) and lines[index].strip().startswith("|"):
            rows.append(cells(lines[index]))
            index += 1

        columns = len(header)
        header_style = ParagraphStyle(
            "th", fontName=f"{self.body}-Bold", fontSize=8.4, leading=11.2,
            textColor=colors.HexColor("#0d1117"),
        )
        cell_style = ParagraphStyle(
            "td", fontName=self.body, fontSize=8.4, leading=11.2,
            textColor=colors.HexColor("#22262d"),
        )

        data = [[Paragraph(inline(c, self.mono), header_style) for c in header]]
        for row in rows:
            row = (row + [""] * columns)[:columns]
            data.append([Paragraph(inline(c, self.mono), cell_style) for c in row])

        available = 165 * mm
        table = Table(data, colWidths=[available / columns] * columns, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef0f3")),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d6d9de")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                     [colors.white, colors.HexColor("#fafbfc")]),
                ]
            )
        )
        return index, KeepTogether([Spacer(1, 3), table, Spacer(1, 10)])


def build_pdf(markdown: str, out_path: Path, title: str) -> tuple[Path, set[str]]:
    # Deterministic output. Without this, reportlab stamps the current time into
    # /CreationDate and /ModDate and generates a random document /ID, so
    # rendering the same markdown twice produces two different files. The PDFs
    # are committed, which made that a real problem rather than an aesthetic
    # one: every render dirtied the working tree whether or not anything had
    # changed, so a diff could not distinguish a real edit from a rebuild, and
    # the question "is the committed PDF current?" had no answer short of
    # reading 49 pages. Invariant mode fixes the timestamp and derives the ID
    # from the content, which makes an unchanged document render to identical
    # bytes.
    rl_config.invariant = 1

    body_font, mono_font, charset = register_fonts()
    markdown, replaced = sanitise(markdown, charset)

    renderer = MarkdownRenderer(body_font, mono_font)
    renderer.body = body_font
    story = renderer.convert(markdown)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=22 * mm,
        rightMargin=23 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        title=title,
        author="sagar2907",
    )

    frame = Frame(
        doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="body",
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
    )

    def decorate(canvas, document):
        canvas.saveState()
        canvas.setFont(body_font, 7.6)
        canvas.setFillColor(colors.HexColor("#8b9099"))
        canvas.drawString(doc.leftMargin, 12 * mm, title)
        canvas.drawRightString(A4[0] - doc.rightMargin, 12 * mm, str(document.page))
        canvas.restoreState()

    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=decorate)])
    doc.build(story)
    return out_path, replaced


def verify(pdf_path: Path, image_dir: Path | None = None) -> dict:
    """Rasterise every page and look for the signature of a glyph failure.

    A PDF with missing glyphs renders without error. The two detectable
    symptoms are a page that is essentially blank when it should carry text,
    and a page with an implausible share of pure-black pixels, which is what a
    run of .notdef boxes looks like.
    """
    import pymupdf

    document = pymupdf.open(pdf_path)
    if image_dir:
        image_dir.mkdir(parents=True, exist_ok=True)

    report = {"pages": document.page_count, "blank": [], "suspicious": [], "chars": 0}
    for number, page in enumerate(document, start=1):
        report["chars"] += len(page.get_text())
        pixmap = page.get_pixmap(dpi=72)
        if image_dir:
            pixmap.save(image_dir / f"page-{number:02d}.png")

        samples = pixmap.samples
        stride = pixmap.n
        total = pixmap.width * pixmap.height
        dark = sum(
            1
            for offset in range(0, len(samples), stride * 7)
            if samples[offset] < 60
        )
        sampled = max(1, len(range(0, len(samples), stride * 7)))
        dark_share = dark / sampled

        if len(page.get_text().strip()) < 20:
            report["blank"].append(number)
        # Body text on A4 covers a few percent of the page. Anything above 35%
        # dark is not prose.
        if dark_share > 0.35 and total:
            report["suspicious"].append({"page": number, "dark_share": round(dark_share, 3)})
    document.close()
    return report


def _next_is_part_heading(lines: list[str], start: int) -> bool:
    """Whether the next non-blank line is a level-1 heading."""
    for line in lines[start:]:
        if line.strip():
            return line.strip().startswith("# ")
    return False


def default_output_path(source: Path) -> Path:
    """Where a document renders to when no output is given.

    Named for the project rather than the source file. A PDF is a document that
    leaves the repository -- it gets mailed, downloaded, opened from a folder of
    unrelated files -- and "report.pdf" identifies nothing once it is out there.

    The default also has to match the committed filename. It did not: the
    default was the source name with a .pdf suffix, so the regeneration command
    in the documentation wrote docs/report.pdf while the tracked file was
    docs/Pattern-Ladder-report.pdf. Following the instructions produced an
    untracked file and left the committed PDF stale, with no error to say so.
    This lives in a function rather than inline in main() so that the naming
    rule can be asserted directly instead of inferred from which files happen
    to exist.
    """
    return source.parent / f"Pattern-Ladder-{source.stem}.pdf"


def main() -> int:
    parser = argparse.ArgumentParser()
    # Every document by default, not just the report. There are two committed
    # PDFs and the no-argument form used to render only one of them, so the
    # corrected brief went stale whenever anything regenerated "the PDF" --
    # silently, because rendering the other one succeeded and said so.
    parser.add_argument("source", nargs="*", default=None)
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--images", default=None, help="also write page PNGs here")
    args = parser.parse_args()

    sources = [Path(s) for s in args.source] if args.source else sorted(DOCS.glob("*.md"))
    if not sources:
        print("no documents to render", file=sys.stderr)
        return 1
    if args.output and len(sources) > 1:
        print("-o takes a single source", file=sys.stderr)
        return 1

    failed = False
    for source in sources:
        if not source.is_file():
            print(f"source not found: {source}", file=sys.stderr)
            return 1

        output = Path(args.output) if args.output else default_output_path(source)
        title = args.title or source.stem.replace("-", " ").replace("_", " ").title()

        markdown = source.read_text(encoding="utf-8")
        pdf_path, replaced = build_pdf(markdown, output, title)

        if replaced:
            print("substituted characters the font cannot draw: " + " ".join(sorted(replaced)))

        report = verify(pdf_path, Path(args.images) if args.images else None)
        print(f"wrote {pdf_path}  ({report['pages']} pages, {report['chars']} chars extracted)")
        if report["blank"]:
            print(f"  WARNING blank pages: {report['blank']}")
            failed = True
        if report["suspicious"]:
            print(f"  WARNING unusually dark pages: {report['suspicious']}")
            failed = True
        if not report["blank"] and not report["suspicious"]:
            print("  visual check passed: no blank or suspiciously dark pages")

    # A blank page is a rendering failure, and a render that reports one while
    # exiting zero is a render that will pass unnoticed in a build.
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
