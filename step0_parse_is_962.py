import fitz          # pymupdf — converts PDF pages to images
import ollama
import base64
import os

TARGET_SECTIONS = ["3", "9", "10", "11"]

# ── Helper: render one PDF page → base64 JPEG ─────────────────────────────────
def _page_to_base64(doc, page_num, dpi=200):
    """Render a single PDF page as a JPEG image (base64 encoded)"""
    page = doc[page_num]
    mat = fitz.Matrix(dpi / 72, dpi / 72)   # scale factor from 72 DPI base
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img_bytes = pix.tobytes("jpeg")
    return base64.b64encode(img_bytes).decode("utf-8")


# ── Pass 1: Scan all pages — find which ones have Sections 3,9,10,11 ──────────
def _find_relevant_pages(doc, total_pages):
    """Ask Moondream for each page: does it contain our target sections?"""
    relevant = []
    print(f"  Pass 1 — Scanning {total_pages} pages for Sections 3, 9, 10, 11...")

    for i in range(total_pages):
        img = _page_to_base64(doc, i)

        resp = ollama.chat(
            model="moondream",
            messages=[{
                "role": "user",
                "content": (
                    "Look at this page from IS 962 Indian Standard document. "
                    "Does it contain any of these sections: "
                    "Section 3 (Size of Drawings), "
                    "Section 9 (Line Works / Line Types / Table 9.2), "
                    "Section 10 (Lettering and Dimensioning), or "
                    "Section 11 (Graphical Symbols)? "
                    "Answer YES or NO only."
                ),
                "images": [img]
            }]
        )

        answer = resp["message"]["content"].strip().upper()
        status = "RELEVANT ✓" if "YES" in answer else "skipped"
        print(f"    Page {i+1}/{total_pages}: {status}")

        if "YES" in answer:
            relevant.append(i)

    return relevant


# ── Pass 2: Extract full content from relevant pages ─────────────────────────
def _extract_page_content(doc, page_num):
    """Ask Moondream to extract all content from a single relevant page"""
    img = _page_to_base64(doc, page_num)

    resp = ollama.chat(
        model="moondream",
        messages=[{
            "role": "user",
            "content": (
                "This is a page from IS 962 Indian Standard (architectural drawing rules). "
                "Extract ALL content from this page completely and accurately. Include:\n"
                "- Section number and heading\n"
                "- All paragraph text and rules\n"
                "- All table data: row by row, with line type codes, names, "
                "  thicknesses, and descriptions (especially Table 9.2 if present)\n"
                "- Any lettering height rules or text style rules\n"
                "- Any graphical symbol descriptions or drawing instructions\n"
                "- Sheet size values or dimension rules if present\n\n"
                "Be thorough. Do not summarize — extract the actual content."
            ),
            "images": [img]
        }]
    )

    return resp["message"]["content"].strip()


# ── Public entry point ────────────────────────────────────────────────────────
def extract_is962_context(pdf_path):
    """
    VLM-based IS 962 PDF parser.
    Converts each page to an image and uses Moondream to read
    both text and visual content (tables, line diagrams, symbols).
    """

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"IS 962 PDF not found at: {pdf_path}")

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"  IS_962.pdf opened — {total_pages} pages")

    # Hardcode relevant pages: Page 6 to Page 29 (indices 5 to 28)
    # Ensure we don't exceed total_pages just in case
    start_page = 5
    end_page = min(29, total_pages)
    relevant_pages = list(range(start_page, end_page))

    # Pass 2 — extract content from each relevant page
    print(f"\n  Pass 2 — Extracting content from {len(relevant_pages)} page(s)...")
    extracted_parts = []

    for page_num in relevant_pages:
        print(f"    Extracting page {page_num + 1}...")
        content = _extract_page_content(doc, page_num)
        extracted_parts.append(f"[Page {page_num + 1}]\n{content}")
        print(f"    Done ({len(content)} chars extracted)")

    doc.close()

    # Combine into one context string for the VLM prompt
    context = "IS 962 Indian Standard — Extracted by VLM:\n\n"
    context += "\n\n".join(extracted_parts)

    print(f"\n  IS 962 context ready — {len(context)} total characters")
    return context


if __name__ == "__main__":
    ctx = extract_is962_context("input/IS 962.pdf")
    print("\n--- Preview (first 600 chars) ---")
    print(ctx[:600])