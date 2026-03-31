"""
Agent2: Writer Agent (real implementation)

Bridges PipelineState -> finetuning/agent2_llama.run_agent2 -> DraftArticle.
This keeps the same node contract expected by graph_pipeline.
"""

import importlib.util
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from state import DraftArticle, PipelineState


def _load_agent2_module():
	root = Path(__file__).resolve().parents[1]
	module_path = root / "finetuning" / "agent2_llama.py"

	if not module_path.exists():
		raise FileNotFoundError(f"Agent2 module not found: {module_path}")

	spec = importlib.util.spec_from_file_location("finetuning_agent2_llama", module_path)
	if spec is None or spec.loader is None:
		raise ImportError(f"Failed to load module spec from {module_path}")

	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


def _build_topic_with_revision_context(state: PipelineState) -> str:
	topic = str(state.get("topic") or "Technical Topic")
	revision_count = int(state.get("revision_count", 0) or 0)
	evaluation = state.get("evaluation") or {}

	if revision_count <= 0 or not isinstance(evaluation, dict):
		return topic

	suggestions = evaluation.get("rewrite_suggestions") or []
	remarks = evaluation.get("remarks") or []

	selected = []
	if isinstance(suggestions, list):
		selected.extend([str(x).strip() for x in suggestions[:3] if str(x).strip()])
	if isinstance(remarks, list) and not selected:
		selected.extend([str(x).strip() for x in remarks[:2] if str(x).strip()])

	if not selected:
		return topic

	guidance = " | ".join(selected)
	return f"{topic}\nRevision guidance: {guidance}"


def _build_agent1_payload_from_state(state: PipelineState) -> Dict[str, Any]:
	research_data = state.get("research_data") or {}
	definitions = research_data.get("definitions") if isinstance(research_data, dict) else {}
	sources = research_data.get("sources") if isinstance(research_data, dict) else []

	if not isinstance(definitions, dict):
		definitions = {}
	if not isinstance(sources, list):
		sources = []

	return {
		"research_data": {
			"topic": _build_topic_with_revision_context(state),
			"definitions": definitions,
			"sources": sources,
		}
	}


def _derive_meta_description(body: str) -> str:
	stripped_body = (body or "").strip()
	if not stripped_body:
		return ""

	first_paragraph = stripped_body.split("\n\n", 1)[0].strip()
	first_sentence = re.split(r"(?<=[.!?])\s+", first_paragraph)[0].strip()

	if len(first_sentence) <= 160:
		return first_sentence
	return first_sentence[:157].rstrip() + "..."


def _build_citations_from_ids(used_ids: List[int], state: PipelineState) -> List[str]:
	research_data = state.get("research_data") or {}
	sources = research_data.get("sources", []) if isinstance(research_data, dict) else []

	url_by_id: Dict[int, str] = {}
	if isinstance(sources, list):
		for source in sources:
			if not isinstance(source, dict):
				continue
			sid = source.get("id")
			url = source.get("url")
			if isinstance(sid, int) and isinstance(url, str) and url.strip():
				url_by_id[sid] = url.strip()

	citations: List[str] = []
	for sid in used_ids:
		if sid in url_by_id:
			citations.append(f"[{sid}] {url_by_id[sid]}")
	return citations


def _minimal_fallback_draft(state: PipelineState, error: Exception) -> DraftArticle:
	topic = str(state.get("topic") or "Technical Topic")
	body = (
		f"A writer generation error occurred for topic '{topic}'. "
		f"Please retry this step. Error: {type(error).__name__}: {error}"
	)
	return {
		"title": f"Draft generation failed: {topic}",
		"meta_description": _derive_meta_description(body),
		"content_md": body,
		"citations": [],
	}


def writer_agent_node(state: PipelineState) -> dict:
	"""
	Real Writer Agent implementation.

	INPUT CONTRACT:
		- state["research_data"]: ResearchData
		- state["persona"]: str (default: "Technical Journalist")
		- state["word_count"]: int (default: 800)
		- state["evaluation"]: Optional[Evaluation] (for revisions)

	OUTPUT CONTRACT:
		- DraftArticle with title, meta_description, content_md, citations
	"""
	persona = state.get("persona", "Technical Journalist")
	word_count = state.get("word_count", 800)
	revision_count = state.get("revision_count", 0)

	print(f"[WRITER AGENT] Persona: {persona}, Target: {word_count} words")
	if revision_count > 0 and state.get("evaluation"):
		print(f"[WRITER AGENT] REVISION #{revision_count}")

	try:
		module = _load_agent2_module()
		payload = _build_agent1_payload_from_state(state)

		with tempfile.TemporaryDirectory(prefix="writer_agent_") as temp_dir:
			temp_path = Path(temp_dir)
			agent1_input = temp_path / "agent1_input.json"
			agent2_output = temp_path / "agent2_output.json"

			agent1_input.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

			run_agent2 = getattr(module, "run_agent2")
			result = run_agent2(str(agent1_input), str(agent2_output))

			output_payload = result.get("agent2_output", {}) if isinstance(result, dict) else {}
			title = str(output_payload.get("title") or "Technical Analysis").strip()
			body = str(output_payload.get("body") or "").strip()
			used_ids_raw = output_payload.get("used_source_ids") or []

			used_ids: List[int] = []
			if isinstance(used_ids_raw, list):
				for item in used_ids_raw:
					if isinstance(item, int):
						used_ids.append(item)

			draft_article: DraftArticle = {
				"title": title,
				"meta_description": _derive_meta_description(body),
				"content_md": body,
				"citations": _build_citations_from_ids(sorted(set(used_ids)), state),
			}

			print(f"[WRITER AGENT] Generated article: '{draft_article['title']}'")
			print("[WRITER AGENT] ✓ Checkpoint will be saved after this node")

			return {"draft_article": draft_article}

	except Exception as error:
		print(f"[WRITER AGENT] ERROR: {error}")
		fallback = _minimal_fallback_draft(state, error)
		print("[WRITER AGENT] Returning fallback draft to keep pipeline alive")
		print("[WRITER AGENT] ✓ Checkpoint will be saved after this node")
		return {"draft_article": fallback}

