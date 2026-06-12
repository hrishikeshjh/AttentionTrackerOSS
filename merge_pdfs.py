"""
merge_pdfs.py

Merges README-overall.pdf, README-v1.0.pdf, README-v2.0.pdf
into a single CognEn-Complete-Documentation.pdf with:
  1. A Table of Contents cover page (first page)
  2. Overall documentation (as-is)
  3. v1.0 documentation
  4. v2.0 documentation

Bookmarks/outlines are added for easy navigation.
"""

import io
import struct
import zlib
from pathlib import Path
from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    ArrayObject,
    DictionaryObject,
    FloatObject,
    NameObject,
    NumberObject,
    ByteStringObject,
    BooleanObject,
)


# ---------------------------------------------------------------------------
# Minimal raw-PDF TOC page builder (no external deps)
# ---------------------------------------------------------------------------

def build_toc_pdf(sections: list[dict]) -> bytes:
    """
    Build a minimal single-page PDF containing project details + TOC.
    White background, clean professional layout.
    sections: list of dicts with keys: title, start_page (1-indexed in final doc)
    Returns raw PDF bytes.
    """

    # A4 dimensions: 595.28 x 841.89 pts
    W, H = 595.28, 841.89
    MARGIN = 56.0

    def escape(text):
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    content_parts = []

    # ---- White background ----
    content_parts.append(f"1.000 1.000 1.000 rg 0 0 {W:.2f} {H:.2f} re f")

    # ---- Top header bar (deep navy) ----
    header_h = 72
    content_parts.append(f"0.086 0.122 0.208 rg 0 {H - header_h:.2f} {W:.2f} {header_h:.2f} re f")

    # Thin accent stripe below header bar
    content_parts.append(f"0.180 0.420 0.820 rg 0 {H - header_h - 3:.2f} {W:.2f} 3 re f")

    # ---- "Progress Review Report" in header bar ----
    content_parts.append(
        f"BT /Helvetica-Bold 22 Tf 1.000 1.000 1.000 rg "
        f"{MARGIN:.2f} {H - 46:.2f} Td (Progress Review Report) Tj ET"
    )

    # ---- Project details block ----
    # Positions (y descends from top)
    details_top = H - header_h - 3 - 36   # first detail line

    # Labels in muted grey, values in dark ink
    label_r, label_g, label_b = 0.40, 0.42, 0.48
    value_r, value_g, value_b = 0.08, 0.09, 0.12
    accent_r, accent_g, accent_b = 0.18, 0.42, 0.82

    rows = [
        ("Proposal Title", "A System for estimating participants engagement by analyzing"),
        ("",               "attention and multitasking dynamics"),
        ("Project Code",   "TIH/RP/20"),
        ("PI and Co-PIs",  "Prof. Samiran Chattopadhyay & Prof. Dr. Mourani Sinha"),
        ("Project Tenure", "Completed"),
    ]

    line_h = 19
    y_cur  = details_top
    label_col = MARGIN
    value_col = MARGIN + 118

    for label, value in rows:
        if label:
            # Label
            content_parts.append(
                f"BT /Helvetica-Bold 10 Tf {label_r:.3f} {label_g:.3f} {label_b:.3f} rg "
                f"{label_col:.2f} {y_cur:.2f} Td ({escape(label)}) Tj ET"
            )
            # Colon separator
            content_parts.append(
                f"BT /Helvetica 10 Tf {label_r:.3f} {label_g:.3f} {label_b:.3f} rg "
                f"{label_col + 108:.2f} {y_cur:.2f} Td (:) Tj ET"
            )
        # Value
        content_parts.append(
            f"BT /Helvetica 10 Tf {value_r:.3f} {value_g:.3f} {value_b:.3f} rg "
            f"{value_col:.2f} {y_cur:.2f} Td ({escape(value)}) Tj ET"
        )
        y_cur -= line_h

    # Horizontal rule under project details
    rule_y = y_cur - 10
    content_parts.append(
        f"0.820 0.840 0.880 rg {MARGIN:.2f} {rule_y:.2f} {W - 2*MARGIN:.2f} 0.75 re f"
    )

    # ---- TABLE OF CONTENTS header ----
    toc_header_y = rule_y - 30
    content_parts.append(
        f"BT /Helvetica-Bold 11 Tf {accent_r:.3f} {accent_g:.3f} {accent_b:.3f} rg "
        f"{MARGIN:.2f} {toc_header_y:.2f} Td (TABLE OF CONTENTS) Tj ET"
    )

    # Small underline under TOC header
    content_parts.append(
        f"{accent_r:.3f} {accent_g:.3f} {accent_b:.3f} rg "
        f"{MARGIN:.2f} {toc_header_y - 4:.2f} 120 1.2 re f"
    )

    # ---- Section rows ----
    section_colors = [
        (0.18, 0.42, 0.82),   # blue
        (0.10, 0.62, 0.48),   # teal
        (0.62, 0.28, 0.84),   # purple
    ]
    section_numbers = ["01", "02", "03"]

    row_start_y = toc_header_y - 26
    card_h = 38
    card_gap = 10

    for i, section in enumerate(sections):
        y = row_start_y - i * (card_h + card_gap)
        cr, cg, cb = section_colors[i]

        # Card background (very light grey)
        content_parts.append(
            f"0.960 0.965 0.975 rg {MARGIN:.2f} {y:.2f} {W - 2*MARGIN:.2f} {card_h:.2f} re f"
        )
        # Left accent stripe
        content_parts.append(
            f"{cr:.3f} {cg:.3f} {cb:.3f} rg {MARGIN:.2f} {y:.2f} 4 {card_h:.2f} re f"
        )

        # Number badge
        content_parts.append(
            f"BT /Helvetica-Bold 10 Tf {cr:.3f} {cg:.3f} {cb:.3f} rg "
            f"{MARGIN + 12:.2f} {y + card_h - 15:.2f} Td ({escape(section_numbers[i])}) Tj ET"
        )

        # Section title
        content_parts.append(
            f"BT /Helvetica-Bold 11 Tf {value_r:.3f} {value_g:.3f} {value_b:.3f} rg "
            f"{MARGIN + 32:.2f} {y + card_h - 14:.2f} Td ({escape(section['title'])}) Tj ET"
        )

        # Dotted line
        content_parts.append(
            f"BT /Helvetica 9 Tf 0.65 0.68 0.74 rg "
            f"{MARGIN + 32:.2f} {y + card_h - 27:.2f} Td "
            f"(. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .) Tj ET"
        )

        # Page number
        pg_text = f"Page {section['start_page']}"
        content_parts.append(
            f"BT /Helvetica-Bold 10 Tf {label_r:.3f} {label_g:.3f} {label_b:.3f} rg "
            f"{W - MARGIN - 52:.2f} {y + card_h - 14:.2f} Td ({escape(pg_text)}) Tj ET"
        )

    # ---- Bottom footer ----
    footer_y = 28
    content_parts.append(
        f"0.820 0.840 0.880 rg {MARGIN:.2f} {footer_y + 12:.2f} {W - 2*MARGIN:.2f} 0.5 re f"
    )
    content_parts.append(
        f"BT /Helvetica 8 Tf 0.55 0.58 0.64 rg "
        f"{MARGIN:.2f} {footer_y:.2f} Td "
        f"(All content is preserved as-is from the individual version documents.) Tj ET"
    )

    content_stream = "\n".join(content_parts).encode("latin-1")

    # Build raw PDF manually (minimal but valid)
    objects = {}

    # Object 1: Catalog
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"

    # Object 2: Pages
    objects[2] = b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"

    # Object 3: Page
    objects[3] = (
        f"<< /Type /Page /Parent 2 0 R "
        f"/MediaBox [0 0 {W:.2f} {H:.2f}] "
        f"/Contents 4 0 R "
        f"/Resources << /Font << "
        f"/Helvetica << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> "
        f"/Helvetica-Bold << /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >> "
        f">> >> >>"
    ).encode()

    # Object 4: Content stream
    stream_bytes = content_stream
    objects[4] = f"<< /Length {len(stream_bytes)} >>".encode() + b"\nstream\n" + stream_bytes + b"\nendstream"

    # Assemble PDF
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = {}
    for oid, obj_data in objects.items():
        offsets[oid] = out.tell()
        out.write(f"{oid} 0 obj\n".encode())
        out.write(obj_data)
        out.write(b"\nendobj\n")

    xref_pos = out.tell()
    out.write(b"xref\n")
    out.write(f"0 {len(objects) + 1}\n".encode())
    out.write(b"0000000000 65535 f \n")
    for oid in sorted(objects.keys()):
        out.write(f"{offsets[oid]:010d} 00000 n \n".encode())

    out.write(b"trailer\n")
    out.write(f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode())
    out.write(b"startxref\n")
    out.write(f"{xref_pos}\n".encode())
    out.write(b"%%EOF\n")

    return out.getvalue()


# ---------------------------------------------------------------------------
# Page number stamp builder
# ---------------------------------------------------------------------------

def build_page_number_stamp(page_num: int, total_pages: int, W: float, H: float) -> bytes:
    """
    Build a minimal transparent single-page PDF with a small page number
    at the top-right corner. Intended to be overlaid onto existing pages.
    """

    def escape(text):
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    text = f"{page_num} / {total_pages}"

    # Small pill background at top-right
    pill_w, pill_h = 58, 16
    pill_x = W - pill_w - 14
    pill_y = H - pill_h - 10

    content_parts = [
        # Light grey pill background
        f"0.920 0.925 0.935 rg {pill_x:.2f} {pill_y:.2f} {pill_w:.2f} {pill_h:.2f} re f",
        # Page number text, dark ink, centered in pill
        f"BT /Helvetica 8 Tf 0.30 0.32 0.38 rg "
        f"{pill_x + 8:.2f} {pill_y + 4:.2f} Td ({escape(text)}) Tj ET",
    ]

    content_stream = "\n".join(content_parts).encode("latin-1")

    objects = {}
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[2] = b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"
    objects[3] = (
        f"<< /Type /Page /Parent 2 0 R "
        f"/MediaBox [0 0 {W:.2f} {H:.2f}] "
        f"/Contents 4 0 R "
        f"/Resources << /Font << "
        f"/Helvetica << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> "
        f">> >> >>"
    ).encode()
    objects[4] = (
        f"<< /Length {len(content_stream)} >>".encode()
        + b"\nstream\n" + content_stream + b"\nendstream"
    )

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = {}
    for oid, obj_data in objects.items():
        offsets[oid] = out.tell()
        out.write(f"{oid} 0 obj\n".encode())
        out.write(obj_data)
        out.write(b"\nendobj\n")

    xref_pos = out.tell()
    out.write(b"xref\n")
    out.write(f"0 {len(objects) + 1}\n".encode())
    out.write(b"0000000000 65535 f \n")
    for oid in sorted(objects.keys()):
        out.write(f"{offsets[oid]:010d} 00000 n \n".encode())
    out.write(b"trailer\n")
    out.write(f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode())
    out.write(b"startxref\n")
    out.write(f"{xref_pos}\n".encode())
    out.write(b"%%EOF\n")

    return out.getvalue()


# ---------------------------------------------------------------------------
# Main merge logic
# ---------------------------------------------------------------------------

def main():
    base = Path(__file__).parent

    pdf_overall = base / "README-overall.pdf"
    pdf_v1      = base / "README-v1.0.pdf"
    pdf_v2      = base / "README-v2.0.pdf"
    output_pdf  = base / "CognEn-Complete-Documentation.pdf"

    # Read page counts
    reader_overall = PdfReader(pdf_overall)
    reader_v1      = PdfReader(pdf_v1)
    reader_v2      = PdfReader(pdf_v2)

    pages_overall = len(reader_overall.pages)
    pages_v1      = len(reader_v1.pages)
    pages_v2      = len(reader_v2.pages)

    print(f"Overall: {pages_overall} pages")
    print(f"v1.0:    {pages_v1} pages")
    print(f"v2.0:    {pages_v2} pages")

    # TOC page will be page 1.
    # Overall starts at page 2 (1-indexed).
    # v1.0 starts after overall.
    # v2.0 starts after v1.0.
    start_overall = 2
    start_v1      = start_overall + pages_overall
    start_v2      = start_v1 + pages_v1

    sections = [
        {"title": "Overall Documentation",     "start_page": start_overall},
        {"title": "Version 1.0 Documentation", "start_page": start_v1},
        {"title": "Version 2.0 Documentation", "start_page": start_v2},
    ]

    print("\nBuilding TOC page...")
    toc_bytes = build_toc_pdf(sections)
    toc_reader = PdfReader(io.BytesIO(toc_bytes))

    print("Merging PDFs...")
    writer = PdfWriter()

    # Collect all source pages in order (TOC first, then each section)
    all_source_pages = (
        [toc_reader.pages[0]]
        + list(reader_overall.pages)
        + list(reader_v1.pages)
        + list(reader_v2.pages)
    )
    total_pages = len(all_source_pages)

    print(f"Stamping page numbers ({total_pages} pages)...")
    for i, src_page in enumerate(all_source_pages):
        page_num = i + 1  # 1-indexed

        # Determine page dimensions (fall back to A4 if not set)
        mb = src_page.mediabox
        pw = float(mb.width)  if mb else 595.28
        ph = float(mb.height) if mb else 841.89

        # Build stamp and merge onto the page
        stamp_bytes = build_page_number_stamp(page_num, total_pages, pw, ph)
        stamp_page  = PdfReader(io.BytesIO(stamp_bytes)).pages[0]
        src_page.merge_page(stamp_page)

        writer.add_page(src_page)

    # Add named bookmarks/outlines
    writer.add_outline_item("Table of Contents",         0)
    writer.add_outline_item("Overall Documentation",     1)
    writer.add_outline_item("Version 1.0 Documentation", 1 + pages_overall)
    writer.add_outline_item("Version 2.0 Documentation", 1 + pages_overall + pages_v1)

    # Set PDF metadata
    writer.add_metadata({
        "/Title":   "CognEn-VCSir Complete Documentation",
        "/Author":  "CognEn-VCSir",
        "/Subject": "Overall + v1.0 + v2.0 Documentation",
        "/Creator": "merge_pdfs.py",
    })

    with open(output_pdf, "wb") as f:
        writer.write(f)

    total = 1 + pages_overall + pages_v1 + pages_v2
    print(f"\n✅ Done! Merged PDF saved to: {output_pdf}")
    print(f"   Total pages: {total}")
    print(f"   Page 1          → Table of Contents")
    print(f"   Pages 2–{start_v1 - 1:<4}   → Overall Documentation ({pages_overall} pages)")
    print(f"   Pages {start_v1}–{start_v2 - 1:<4}  → v1.0 Documentation ({pages_v1} pages)")
    print(f"   Pages {start_v2}–{total:<4}  → v2.0 Documentation ({pages_v2} pages)")


if __name__ == "__main__":
    main()
