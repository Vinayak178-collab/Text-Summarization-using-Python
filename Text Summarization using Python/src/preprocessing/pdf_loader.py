from __future__ import annotations

from io import BytesIO, StringIO
from typing import Optional

from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams


def extract_text_from_pdf_bytes(data: bytes, max_pages: Optional[int] = None) -> str:
    """
    Extract text from a PDF given its raw bytes.

    Parameters
    ----------
    data : bytes
        PDF file content.
    max_pages : Optional[int]
        Optional limit on number of pages to process for speed.
    """
    input_fp = BytesIO(data)
    output = StringIO()
    laparams = LAParams()
    extract_text_to_fp(
        fp=input_fp,
        outfp=output,
        laparams=laparams,
        maxpages=max_pages or 0,
    )
    return output.getvalue()


