import json
import re
from pathlib import Path
from typing import Any

from state import DraftArticle, ResearchData


def _read_json(path: str) -> Any:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_research_data_from_file(path: str) -> ResearchData:
    data = _read_json(path)

    if "research_data" in data:
        research_data = data["research_data"]
    else:
        research_data = data

    if not isinstance(research_data, dict):
        raise ValueError("Invalid Agent1 JSON: expected object for research_data")

    definitions = research_data.get("definitions", {})
    sources = research_data.get("sources", [])

    if not isinstance(definitions, dict):
        raise ValueError("Invalid Agent1 JSON: research_data.definitions must be an object")
    if not isinstance(sources, list):
        raise ValueError("Invalid Agent1 JSON: research_data.sources must be a list")

    return {
        "definitions": definitions,
        "sources": sources,
    }


def _derive_meta_description(body: str) -> str:
    stripped_body = body.strip()
    if not stripped_body:
        return ""

    first_paragraph = stripped_body.split("\n\n", 1)[0].strip()
    first_sentence = re.split(r"(?<=[.!?])\s+", first_paragraph)[0].strip()

    if len(first_sentence) <= 160:
        return first_sentence
    return first_sentence[:157].rstrip() + "..."


def _extract_citations(body: str) -> list[str]:
    citations = re.findall(r"\[(\d+)\]\s*<([^>]+)>", body)
    return [f"[{index}] {url.strip()}" for index, url in citations]


def load_draft_article_from_file(path: str) -> DraftArticle:
    data = _read_json(path)

    if "agent2_output" in data:
        draft_payload = data["agent2_output"]
    else:
        draft_payload = data

    if not isinstance(draft_payload, dict):
        raise ValueError("Invalid Agent2 JSON: expected object for agent2_output")

    title = draft_payload.get("title")
    body = draft_payload.get("body") or draft_payload.get("content_md")

    if not title or not isinstance(title, str):
        raise ValueError("Invalid Agent2 JSON: title is required")
    if not body or not isinstance(body, str):
        raise ValueError("Invalid Agent2 JSON: body/content_md is required")

    meta_description = draft_payload.get("meta_description")
    if not meta_description:
        meta_description = _derive_meta_description(body)

    citations = draft_payload.get("citations")
    if not isinstance(citations, list):
        citations = _extract_citations(body)

    return {
        "title": title,
        "meta_description": meta_description,
        "content_md": body,
        "citations": citations,
    }
