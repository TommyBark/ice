from langchain_community.document_loaders import PDFMinerPDFasHTMLLoader
from typing import Literal
from bs4 import BeautifulSoup
import re
from langchain_community.docstore.document import Document
from ice.paper import Paragraph, Section, Paper
from ice.cache import diskcache
from pathlib import Path

SectionType = Literal["abstract", "main", "back"]


@diskcache
def parse_pdf(pdf_path: Path) -> Paper:
    loader = PDFMinerPDFasHTMLLoader(pdf_path)
    data = loader.load()[0]

    # Parse the HTML content
    soup = BeautifulSoup(data.page_content, "html.parser")
    content = soup.find_all("div")
    snippets = _get_pdf_snippets(content)
    semantic_snippets = _get_semantic_snippets(snippets)
    paper = snippet_to_paragraph_parser(semantic_snippets)
    return paper


def _get_pdf_snippets(content: list) -> list:
    # Taken from langchain docs

    cur_fs = None
    cur_text = ""
    snippets = []  # first collect all snippets that have the same font size
    for c in content:
        sp = c.find("span")
        if not sp:
            continue
        st = sp.get("style")
        if not st:
            continue
        fs = re.findall(r"font-size:(\d+)px", st)
        if not fs:
            continue
        fs = int(fs[0])
        if not cur_fs:
            cur_fs = fs
        if fs == cur_fs:
            cur_text += c.text
        else:
            snippets.append((cur_text, cur_fs))
            cur_fs = fs
            cur_text = c.text
    snippets.append((cur_text, cur_fs))
    return snippets


def _get_semantic_snippets(snippets: list) -> list:

    # Taken from langchain docs
    cur_idx = -1
    semantic_snippets = []
    # Assumption: headings have higher font size than their respective content
    for s in snippets:
        # if current snippet's font size > previous section's heading => it is a new heading
        if (
            not semantic_snippets
            or s[1] > semantic_snippets[cur_idx].metadata["heading_font"]
        ):
            metadata = {"heading": s[0], "content_font": 0, "heading_font": s[1]}
            semantic_snippets.append(Document(page_content="", metadata=metadata))
            cur_idx += 1
            continue

        # if current snippet's font size <= previous section's content => content belongs to the same section (one can also create
        # a tree like structure for sub sections if needed but that may require some more thinking and may be data specific)
        if (
            not semantic_snippets[cur_idx].metadata["content_font"]
            or s[1] <= semantic_snippets[cur_idx].metadata["content_font"]
        ):
            semantic_snippets[cur_idx].page_content += s[0]
            semantic_snippets[cur_idx].metadata["content_font"] = max(
                s[1], semantic_snippets[cur_idx].metadata["content_font"]
            )
            continue

        # if current snippet's font size > previous section's content but less than previous section's heading than also make a new
        # section (e.g. title of a PDF will have the highest font size but we don't want it to subsume all sections)
        metadata = {"heading": s[0], "content_font": 0, "heading_font": s[1]}
        semantic_snippets.append(Document(page_content="", metadata=metadata))
        cur_idx += 1
    return semantic_snippets


def section_type_parser(section_str: str) -> SectionType:
    if "abstract" in section_str.lower():
        return "abstract"
    if "references" in section_str.lower():
        return "back"
    else:
        return "main"


def snippet_to_paragraph_parser(snippets: list) -> Paper:
    paragraphs = []
    for snippet in snippets:
        sections = [Section(title=snippet.metadata["heading"])]
        section_type = section_type_parser(snippet.metadata["heading"])

        # This is created for Pylance to be happy
        data = {
            "sentences": [snippet.page_content],
            "sections": sections,
            "sectionType": section_type,
        }
        par = Paragraph(**data)
        paragraphs.append(par)
    return Paper(paragraphs=paragraphs)
