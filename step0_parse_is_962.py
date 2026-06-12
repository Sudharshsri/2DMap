import fitz
import os


def _extract_page_content(doc, page_num):
    """Extract text directly from a PDF page using PyMuPDF."""
    page = doc[page_num]
    text = page.get_text()          # reads embedded text layer — no AI needed
    return text.strip()


def extract_is962_context(pdf_path):
    """
    Direct PyMuPDF text extraction from IS 962 PDF.
    Reads pages 6-29 (indices 5-28) which contain:
      - Section 3  : Sheet sizes (Table 1 — A0 to A4 dimensions)
      - Section 9  : Line work (Table 5 — line type codes, names, thicknesses)
      - Section 10 : Lettering and dimensioning rules
      - Section 11 : Graphical symbols (door/window descriptions)
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"IS 962 PDF not found at: {pdf_path}")

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"  IS_962.pdf opened — {total_pages} pages")

    start_page = 5
    end_page   = min(29, total_pages)
    pages_to_read = list(range(start_page, end_page))

    print(f"  Extracting text from pages {start_page + 1} to {end_page}...")

    extracted_parts = []
    for page_num in pages_to_read:
        content = _extract_page_content(doc, page_num)
        extracted_parts.append(f"[Page {page_num + 1}]\n{content}")
        print(f"    Page {page_num + 1}: {len(content)} chars extracted")

    doc.close()

    context  = "IS 962 Indian Standard — Extracted by PyMuPDF:\n\n"
    context += "\n\n".join(extracted_parts)

    print(f"\n  IS 962 context ready — {len(context)} total characters")
    return context


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    ctx = extract_is962_context("input/IS 962.pdf")
    with open("output/is962_context.txt", "w", encoding="utf-8") as f:
        f.write(ctx)
    print("\n✅ Saved parsed PDF context to output/is962_context.txt")