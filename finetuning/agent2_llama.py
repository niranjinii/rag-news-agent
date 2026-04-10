import os
import re
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
import requests
from typing import Dict, List, Tuple, Set

# =========================================================
# Agent 2 Final (JSON-first + Ollama backend, upgraded prompts)
# Input : agent1_output.json
# Output: agent2_output.json
# =========================================================


def load_local_env() -> None:
    script_dir = Path(__file__).resolve().parent
    candidates = [script_dir / ".env", script_dir.parent / ".env"]

    for env_path in candidates:
        if not env_path.exists():
            continue

        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value
        break


load_local_env()

# -------------------------
# Config
# -------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

GEN_TEMPERATURE = 0.08
GEN_TOP_P = 0.9
GEN_MAX_TOKENS = 3600

POLISH_TEMPERATURE = 0.08
POLISH_TOP_P = 0.9
POLISH_MAX_TOKENS = 2400

MIN_WORDS = 400
MAX_WORDS = 800

TARGET_MIN_GEN_WORDS = 620
MAX_RETRIES = 3
TARGET_AVG_SENTENCE_WORDS = 24
MAX_SENTENCE_WORDS = 38

FILLER_PHRASES = [
    "significant improvements",
    "major leap",
    "enhanced performance",
    "overall",
    "attractive option",
    "blazing speed",
    "game changer",
]

# -------------------------
# I/O
# -------------------------
def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# -------------------------
# Text utils
# -------------------------
def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))

def extract_citation_ids(text: str) -> Set[int]:
    return {int(x) for x in re.findall(r"\[(\d+)\]", text)}

def normalize_inline_citations(text: str, allowed_ids: Set[int] | None = None) -> str:
    """Normalize citation tokens like [1,2, 3] -> [1] [2] [3] and filter to allowed IDs."""
    if not text:
        return ""

    def _replace(match: re.Match) -> str:
        raw = match.group(1)
        parts = re.split(r"[^0-9]+", raw)
        ids: List[int] = []
        seen: Set[int] = set()
        for part in parts:
            if not part:
                continue
            value = int(part)
            if allowed_ids is not None and value not in allowed_ids:
                continue
            if value in seen:
                continue
            seen.add(value)
            ids.append(value)

        if not ids:
            return ""
        return " ".join(f"[{cid}]" for cid in ids)

    return re.sub(r"\[([0-9,;\s]+)\]", _replace, text)

def sanitize_body(text: str, allowed_ids: Set[int] | None = None) -> str:
    text = normalize_inline_citations(text, allowed_ids=allowed_ids)
    text = re.sub(r"(?im)^\s*(references|bibliography|sources)\s*:?\s*$", "", text)
    text = re.sub(r"https?://\S+", "", text)

    # remove leaked metadata line inside body
    text = re.sub(r"(?im)^\s*used[_\s]*source[_\s]*ids\s*:\s*\[[^\]]*\]\s*$", "", text)

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def normalize_punctuation_spacing(text: str) -> str:
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?])(\S)", r"\1 \2", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def repair_fragmented_numbers(text: str) -> str:
    """Repair common numeric fragmentation artifacts (e.g., '9. 6' -> '9.6')."""
    if not text:
        return text

    fixed = text
    fixed = re.sub(r"(\d)\.\s+(\d)", r"\1.\2", fixed)
    fixed = re.sub(r"(\d)\s+\.\s+(\d)", r"\1.\2", fixed)
    fixed = re.sub(r"(\d+)\.\s+\[(\d+)\]\s+(\d+)", r"\1.\3 [\2]", fixed)
    fixed = re.sub(r"\b(\d+)\s+,\s*(Gbps|Mbps|MHz|GHz)\b", r"\1 \2", fixed, flags=re.IGNORECASE)
    return fixed

def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]

def normalize_tokens(s: str) -> Set[str]:
    stop = {"the", "a", "an", "and", "or", "of", "to", "in", "for", "with", "is", "are", "on", "that", "this", "as", "it", "by"}
    toks = re.findall(r"\b[a-z0-9]+\b", s.lower())
    return {t for t in toks if t not in stop}

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))

def repeated_ngram_ratio(text: str, n: int = 4) -> float:
    tokens = re.findall(r"\b\w+\b", text.lower())
    if len(tokens) < n + 1:
        return 0.0
    grams = [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    return 1 - (len(set(grams)) / len(grams))

def clamp_to_max_words(text: str, max_words: int = MAX_WORDS) -> str:
    if word_count(text) <= max_words:
        return text
    sents = split_sentences(text)
    out = []
    for s in sents:
        out.append(s)
        if word_count(" ".join(out)) >= max_words:
            break
    return " ".join(out).strip()

def filler_count(text: str) -> int:
    low = text.lower()
    return sum(low.count(p) for p in FILLER_PHRASES)

# -------------------------
# Agent1 parsing + source dedupe helpers
# -------------------------
def parse_agent1(agent1_data: Dict) -> Tuple[str, Dict[str, str], List[Dict], List[str]]:
    research_data = agent1_data.get("research_data", {})
    topic = research_data.get("topic", "Technical Topic")
    definitions = research_data.get("definitions", {})
    sources = research_data.get("sources", [])
    revision_feedback = research_data.get("revision_feedback", [])

    if not sources:
        raise ValueError("No sources found in research_data.sources")

    for s in sources:
        if "id" not in s:
            raise ValueError("Each source must contain 'id'")
        sid = int(s["id"])
        s["id"] = sid

    if not isinstance(revision_feedback, list):
        revision_feedback = []

    cleaned_feedback = [str(item).strip() for item in revision_feedback if str(item).strip()]
    return topic, definitions, sources, cleaned_feedback

def normalize_claim(text: str) -> str:
    t = re.sub(r"\s+", " ", text.strip().lower())
    t = re.sub(r"[^a-z0-9\s]", "", t)
    return t

def build_nonredundant_evidence_view(sources: List[Dict]) -> List[Dict]:
    uniq = []
    seen_claims = []
    for s in sources:
        claim = s.get("extracted_claim", "")
        nclaim = normalize_claim(claim)
        redundant = False
        for c in seen_claims:
            c_set = set(c.split())
            n_set = set(nclaim.split())
            sim = len(c_set & n_set) / max(1, len(c_set | n_set))
            if sim >= 0.78:
                redundant = True
                break
        s2 = dict(s)
        s2["redundant_claim"] = redundant
        uniq.append(s2)
        if nclaim:
            seen_claims.append(nclaim)
    return uniq

# -------------------------
# Prompt builders
# -------------------------
def build_generation_prompt(
    topic: str,
    definitions: Dict[str, str],
    sources: List[Dict],
    min_words_target: int,
    max_words_target: int,
    revision_feedback: List[str] | None = None,
) -> str:
    defs_text = "\n".join([f"- {k}: {v}" for k, v in definitions.items()]) if definitions else "- None"

    evidence = build_nonredundant_evidence_view(sources)
    blocks = []
    for s in evidence:
        redundancy_note = " (overlaps with another source; use for corroboration, not repetition)" if s.get("redundant_claim") else ""
        blocks.append(
            f"[{s['id']}]{redundancy_note}\n"
            f"Subtopic: {s.get('subtopic','')}\n"
            f"Claim: {s.get('extracted_claim','')}\n"
            f"Chunk: {s.get('raw_chunk','')}\n"
        )

    required_ids = sorted({s["id"] for s in sources})
    required_ids_str = ", ".join([str(i) for i in required_ids])

    feedback_lines = "\n".join([f"- {item}" for item in (revision_feedback or [])])
    revision_block = ""
    if feedback_lines:
        revision_block = f"""
REVISION FEEDBACK (HIGH PRIORITY):
{feedback_lines}

REVISION POLICY (MANDATORY):
- Only rewrite claims that are unsupported, ambiguous, or citation-mismatched.
- Preserve already-supported claims and their citation IDs.
- Do not introduce new facts, entities, or numbers not present in evidence.
""".strip()

    return f"""
You are a senior technical analyst. Produce one grounded technical article from the provided evidence only.

TOPIC:
{topic}

DEFINITIONS:
{defs_text}

EVIDENCE:
{chr(10).join(blocks)}

{revision_block}

STRICT FACTUAL RULES:
1) Use ONLY facts present in evidence.
2) Every material claim must have inline citation(s) in [n] format.
3) Use all required IDs at least once: [{required_ids_str}]
4) If claims overlap across sources, synthesize once and cite multiple IDs.
5) No URLs, no references section, no markdown, no bullet lists.

STYLE RULES (MANDATORY):
- Avoid generic filler like: "significant improvements", "major leap", "enhanced performance", "overall".
- Prefer concrete language with numbers, comparisons, and implications.
- Keep tone neutral and technical.
- No hype or marketing wording.
- Use short, clear sentences (target ~14-24 words each).
- Avoid very long sentences with multiple clauses.
- Avoid repeating the same claim in different wording.

ANALYSIS RULES (MANDATORY):
- If the topic/evidence includes multiple products, provide clear, evidence-backed comparison points.
- If the topic/evidence is a single product, focus on architecture/specs, measured performance, and practical usage implications.
- Include at least 2 short "what this means in practice" statements tied to cited facts.
- Include one concise recommendation statement based only on cited evidence.

LENGTH:
- body must be {min_words_target}-{max_words_target} words.

OUTPUT:
Return ONLY valid JSON with exactly these keys:
{{
  "title": "string",
  "body": "string",
  "used_source_ids": [int, int, ...]
}}

JSON REQUIREMENTS:
- Escape newlines inside strings as \\n.
- used_source_ids must contain all required IDs (sorted, unique).
- Preferred compact JSON:
{{"title":"...","body":"...","used_source_ids":[...]}}
""".strip()

def build_polish_prompt(
    title: str,
    body: str,
    source_ids: List[int],
    min_words_target: int,
    max_words_target: int,
    revision_feedback: List[str] | None = None,
) -> str:
    ids = ", ".join([str(i) for i in source_ids])
    feedback_lines = "\n".join([f"- {item}" for item in (revision_feedback or [])])
    revision_block = ""
    if feedback_lines:
        revision_block = f"""
REVISION FEEDBACK (HIGH PRIORITY):
{feedback_lines}

POLISH POLICY (MANDATORY):
- Keep supported sentences unchanged whenever possible.
- Edit only sentences that violate the feedback above.
- Preserve exact citation IDs on already-supported statements.
""".strip()

    return f"""
Polish the JSON content for precision and depth while preserving facts/citations.

INPUT JSON:
{{
  "title": {json.dumps(title, ensure_ascii=False)},
  "body": {json.dumps(body, ensure_ascii=False)},
  "used_source_ids": [{ids}]
}}

MANDATORY EDIT GOALS:
1) Remove repetition and generic filler phrasing.
2) Improve evidence-grounded technical clarity and structure.
3) Keep all numeric facts and citation anchors intact.
4) Keep only citation IDs from [{ids}].
5) Keep neutral technical tone; remove hype.
6) Body length: {min_words_target}-{max_words_target} words.
7) Keep sentences concise; split long multi-clause sentences.

{revision_block}

Return ONLY valid JSON with same 3 keys.
Escape newlines as \\n inside JSON strings.
""".strip()


def source_evidence_tokens(source: Dict) -> Set[str]:
    evidence_text = " ".join([
        str(source.get("subtopic", "")),
        str(source.get("extracted_claim", "")),
        str(source.get("raw_chunk", "")),
    ])
    return normalize_tokens(evidence_text)


def strip_citations(sentence: str) -> str:
    return re.sub(r"\[(?:\d+(?:\s*,\s*\d+)*)\]", "", sentence).strip()


def is_claim_like_sentence(sentence: str) -> bool:
    core = strip_citations(sentence)
    if len(core.split()) < 6:
        return False
    has_number = bool(re.search(r"\d", core))
    technical_kw = bool(re.search(r"\b(latency|throughput|bandwidth|mhz|ghz|qam|mlo|speed|performance|capacity|standard)\b", core, re.IGNORECASE))
    return has_number or technical_kw


def best_source_match(sentence: str, sources: List[Dict]) -> Tuple[int | None, float]:
    sent_tokens = normalize_tokens(strip_citations(sentence))
    if not sent_tokens:
        return None, 0.0

    best_id = None
    best_sim = 0.0
    for src in sources:
        sid = int(src["id"])
        sim = jaccard(sent_tokens, source_evidence_tokens(src))
        if sim > best_sim:
            best_sim = sim
            best_id = sid
    return best_id, best_sim


def remap_citations_by_sentence(body: str, sources: List[Dict], required_ids: List[int]) -> str:
    """Map each claim-like sentence to best source ID and drop invalid citation ids."""
    allowed = set(required_ids)
    sentences = split_sentences(body)
    remapped: List[str] = []

    for sentence in sentences:
        normalized_sentence = normalize_inline_citations(sentence, allowed_ids=allowed)
        if not is_claim_like_sentence(normalized_sentence):
            remapped.append(strip_citations(normalized_sentence))
            continue

        best_id, _ = best_source_match(normalized_sentence, sources)
        core = strip_citations(normalized_sentence)
        if best_id is None:
            remapped.append(core)
        else:
            remapped.append(f"{core} [{best_id}]")

    return sanitize_body(" ".join(remapped), allowed_ids=allowed)


def enforce_evidence_alignment(body: str, sources: List[Dict], required_ids: List[int]) -> str:
    """Prune or rewrite low-support claim-like sentences using nearest source claim."""
    allowed = set(required_ids)
    sentences = split_sentences(body)
    aligned: List[str] = []

    for sentence in sentences:
        clean_sentence = sanitize_body(sentence, allowed_ids=allowed)
        if not clean_sentence:
            continue

        if not is_claim_like_sentence(clean_sentence):
            aligned.append(strip_citations(clean_sentence))
            continue

        best_id, best_sim = best_source_match(clean_sentence, sources)
        if best_id is None:
            continue

        if best_sim < 0.10:
            src = next((item for item in sources if int(item["id"]) == best_id), None)
            replacement = str(src.get("extracted_claim", "")).strip() if src else ""
            if replacement:
                aligned.append(f"{replacement} [{best_id}]")
            continue

        aligned.append(f"{strip_citations(clean_sentence)} [{best_id}]")

    return sanitize_body(" ".join(aligned), allowed_ids=allowed)

# -------------------------
# Ollama call
# -------------------------
def call_ollama(prompt: str, temperature: float, top_p: float, max_tokens: int) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_tokens
        }
    }
    endpoint = f"{OLLAMA_URL}/api/generate"
    start_time = time.perf_counter()
    r = requests.post(endpoint, json=payload, timeout=1800)
    elapsed_seconds = time.perf_counter() - start_time

    print(
        f"[AGENT2][OLLAMA] HTTP {r.status_code} "
        f"model={OLLAMA_MODEL} duration={elapsed_seconds:.2f}s "
        f"endpoint={endpoint}"
    )

    if r.status_code >= 400:
        details = ""
        try:
            details = r.json().get("error", "")
        except Exception:
            details = r.text.strip()

        raise RuntimeError(
            f"Ollama request failed ({r.status_code}) at {endpoint}. "
            f"Model='{OLLAMA_MODEL}'. Details: {details or 'No error details returned.'}"
        )

    data = r.json()
    return data.get("response", "").strip()


def _persist_parse_debug(raw_text: str, stage: str, candidate_json: str = "") -> Path:
    root_dir = Path(__file__).resolve().parents[1]
    log_dir = root_dir / "logs" / "agent2"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S_%fZ")
    file_path = log_dir / f"parse_failure_{stage}_{timestamp}.txt"

    payload = [
        f"stage={stage}",
        f"model={OLLAMA_MODEL}",
        f"ollama_url={OLLAMA_URL}",
        "",
        "=== RAW MODEL OUTPUT ===",
        raw_text,
    ]

    if candidate_json:
        payload.extend([
            "",
            "=== EXTRACTED JSON CANDIDATE ===",
            candidate_json,
        ])

    file_path.write_text("\n".join(payload), encoding="utf-8")
    return file_path

# -------------------------
# Robust JSON parsing
# -------------------------
def parse_json_response(raw_text: str, stage: str = "unknown") -> Dict:
    text = raw_text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        log_path = _persist_parse_debug(raw_text=text, stage=stage)
        raise ValueError(
            f"No JSON object found in model output.\n"
            f"Debug log saved: {log_path}\n"
            f"Preview:\n{text[:800]}"
        )
    obj_text = m.group(0)

    try:
        return json.loads(obj_text)
    except Exception:
        pass

    sanitized = obj_text
    sanitized = sanitized.replace("\r", "\\r").replace("\t", "\\t")
    sanitized = re.sub(r",\s*([}\]])", r"\1", sanitized)

    out = []
    in_string = False
    escape = False
    for ch in sanitized:
        if in_string:
            if escape:
                out.append(ch)
                escape = False
            else:
                if ch == "\\":
                    out.append(ch)
                    escape = True
                elif ch == '"':
                    out.append(ch)
                    in_string = False
                elif ch == "\n":
                    out.append("\\n")
                else:
                    out.append(ch)
        else:
            out.append(ch)
            if ch == '"':
                in_string = True
    sanitized = "".join(out)

    try:
        return json.loads(sanitized)
    except Exception as e:
        log_path = _persist_parse_debug(raw_text=raw_text, stage=stage, candidate_json=obj_text)
        print("---- RAW MODEL OUTPUT (first 1200 chars) ----")
        print(raw_text[:1200])
        print("---- EXTRACTED JSON CANDIDATE (first 1200 chars) ----")
        print(obj_text[:1200])
        raise ValueError(f"Failed to parse model JSON after sanitization: {e}. Debug log saved: {log_path}")

# -------------------------
# Citation repair (soft)
# -------------------------
def inject_missing_citations(body: str, missing_ids: List[int], sources: List[Dict]) -> str:
    out = body
    for mid in missing_ids:
        src = next((s for s in sources if s["id"] == mid), None)
        claim = (src.get("extracted_claim", "").strip() if src else "")
        subtopic = (src.get("subtopic", "").strip() if src else "")
        injected = False

        if claim:
            idx = out.lower().find(claim.lower())
            if idx != -1:
                pos = idx + len(claim)
                out = out[:pos] + f" [{mid}]" + out[pos:]
                injected = True

        if not injected and claim:
            short_phrase = " ".join(claim.split()[:6]).strip()
            if len(short_phrase) > 10:
                idx = out.lower().find(short_phrase.lower())
                if idx != -1:
                    pos = idx + len(short_phrase)
                    out = out[:pos] + f" [{mid}]" + out[pos:]
                    injected = True

        if not injected and subtopic:
            kws = [w for w in re.split(r"[^a-zA-Z0-9]+", subtopic) if len(w) > 4][:3]
            for kw in kws:
                idx = out.lower().find(kw.lower())
                if idx != -1:
                    pos = idx + len(kw)
                    out = out[:pos] + f" [{mid}]" + out[pos:]
                    injected = True
                    break

        if not injected:
            fallback = claim if claim else "This point is supported by provided evidence"
            out += f"\n\n{fallback} [{mid}]."

    return sanitize_body(out)

# -------------------------
# Deterministic trim
# -------------------------
def is_hype_or_speculative(sentence: str) -> bool:
    patterns = [
        r"\blikely to\b",
        r"\bmajor step forward\b",
        r"\bsignificant impact on the tech industry\b",
        r"\bexciting development\b",
        r"\bblazing speed\b",
        r"\battractive option\b",
    ]
    return any(re.search(p, sentence, flags=re.IGNORECASE) for p in patterns)

def deterministic_trim(body: str) -> str:
    sents = split_sentences(body)
    kept = []
    kept_norm = []

    for s in sents:
        if is_hype_or_speculative(s):
            continue

        s_norm = normalize_tokens(s)
        duplicate = False
        for kn in kept_norm:
            if jaccard(s_norm, kn) >= 0.86:
                duplicate = True
                break
        if not duplicate:
            kept.append(s)
            kept_norm.append(s_norm)

    out = " ".join(kept).strip()
    out = sanitize_body(out)
    out = clamp_to_max_words(out, MAX_WORDS)
    return out


def length_policy(sources: List[Dict]) -> Tuple[int, int, int]:
    """Adaptive length range based on distinct source coverage.

    Keeps output in 400-800 range, but avoids forcing verbose repetition when
    evidence is sparse.
    """
    unique_source_ids = {int(s.get("id", 0)) for s in sources if isinstance(s, dict) and "id" in s}
    source_count = len(unique_source_ids)

    if source_count <= 3:
        min_words, max_words = 400, 620
    elif source_count <= 6:
        min_words, max_words = 460, 740
    else:
        min_words, max_words = 520, 800

    target_min = min(max_words - 80, max(min_words, int((min_words + max_words) / 2)))
    return min_words, max_words, target_min


def _sentence_word_counts(text: str) -> List[int]:
    return [word_count(sentence) for sentence in split_sentences(text)]


def readability_metrics(text: str) -> Dict[str, float]:
    counts = _sentence_word_counts(text)
    if not counts:
        return {
            "avg_sentence_words": 0.0,
            "max_sentence_words": 0.0,
            "long_sentence_ratio": 0.0,
        }

    avg_words = sum(counts) / len(counts)
    max_words = max(counts)
    long_ratio = sum(1 for c in counts if c > MAX_SENTENCE_WORDS) / len(counts)
    return {
        "avg_sentence_words": float(avg_words),
        "max_sentence_words": float(max_words),
        "long_sentence_ratio": float(long_ratio),
    }


def _split_by_words(text: str, max_words: int) -> List[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]

    comma_parts = [part.strip() for part in re.split(r",\s+", text) if part.strip()]
    if len(comma_parts) <= 1:
        return [text]

    chunks: List[str] = []
    current_parts: List[str] = []
    current_words = 0
    for part in comma_parts:
        part_words = word_count(part)
        if current_parts and (current_words + part_words) > max_words:
            chunks.append(", ".join(current_parts).strip())
            current_parts = [part]
            current_words = part_words
        else:
            current_parts.append(part)
            current_words += part_words

    if current_parts:
        chunks.append(", ".join(current_parts).strip())

    return [chunk for chunk in chunks if chunk]


def _sentence_citations(sentence: str) -> List[str]:
    citations = re.findall(r"\[\d+\]", sentence)
    deduped: List[str] = []
    seen: Set[str] = set()
    for citation in citations:
        if citation in seen:
            continue
        seen.add(citation)
        deduped.append(citation)
    return deduped


def _format_sentence(core: str, citations: List[str]) -> str:
    text = re.sub(r"\s+", " ", core).strip(" ,;:-")
    text = normalize_punctuation_spacing(text)
    if not text:
        return ""

    text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
    if not re.search(r"[.!?]$", text):
        text += "."

    if citations:
        text = f"{text} {' '.join(citations)}"
    return text


def _is_orphan_fragment(text: str) -> bool:
    raw = text.strip()
    if not raw:
        return True
    low = raw.lower()
    starts_fragment = re.match(r"^(and|or|but|which|that|because|while|whereas|however|therefore)\b", low) is not None
    too_short = word_count(raw) <= 5
    no_verb = re.search(r"\b(is|are|was|were|has|have|supports?|offers?|enables?|improves?|reduces?)\b", low) is None
    return starts_fragment or (too_short and no_verb)


def _merge_or_drop_orphan_fragments(sentences: List[str]) -> List[str]:
    if not sentences:
        return sentences

    merged: List[str] = []
    for sentence in sentences:
        if _is_orphan_fragment(sentence):
            if merged:
                prev = merged[-1].rstrip()
                prev = re.sub(r"[.!?]+$", "", prev).strip()
                merged[-1] = f"{prev}, {sentence.strip()}"
            continue
        merged.append(sentence)

    return [item for item in merged if item.strip()]


def dedupe_sentences(body: str, threshold: float = 0.72) -> str:
    """Drop near-duplicate sentences to improve readability and flow."""
    kept: List[str] = []
    kept_norm: List[Set[str]] = []

    for sentence in split_sentences(body):
        core = strip_citations(sentence)
        norm = normalize_tokens(core)
        if not norm:
            continue

        duplicate = any(jaccard(norm, existing) >= threshold for existing in kept_norm)
        if duplicate:
            continue

        kept.append(sentence)
        kept_norm.append(norm)

    return " ".join(kept).strip()


def dedupe_by_claim_and_citation(body: str) -> str:
    """Remove repeated claim+citation pairs that survive earlier passes."""
    seen: Set[Tuple[str, str]] = set()
    out: List[str] = []

    for sentence in split_sentences(body):
        core = strip_citations(sentence)
        citation_blob = " ".join(_sentence_citations(sentence))
        key = (normalize_claim(core), citation_blob)
        if not key[0]:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(sentence)

    return " ".join(out).strip()


def _sentence_family_key(sentence: str) -> str:
    """Coarse family key for repeated-claim control."""
    core = strip_citations(sentence)
    tokens = sorted(list(normalize_tokens(core)))
    if not tokens:
        return ""
    return " ".join(tokens[:6])


def _prose_transition(prev_sentence: str, current_sentence: str) -> str:
    prev_has_number = bool(re.search(r"\d", strip_citations(prev_sentence)))
    curr_has_number = bool(re.search(r"\d", strip_citations(current_sentence)))
    if prev_has_number and curr_has_number:
        return "In practical terms,"
    if curr_has_number:
        return "From a performance perspective,"
    return "Operationally,"


def article_style_rewrite(body: str) -> str:
    """Rewrite sentence stream into cleaner article-like flow without losing citations."""
    sentences = split_sentences(repair_fragmented_numbers(normalize_punctuation_spacing(body)))
    if not sentences:
        return body

    arranged: List[str] = []
    seen_families: Set[str] = set()
    consecutive_claims = 0

    for sentence in sentences:
        sentence = normalize_punctuation_spacing(sentence)
        if not sentence:
            continue

        family = _sentence_family_key(sentence)
        if family and family in seen_families:
            continue
        if family:
            seen_families.add(family)

        claim_like = is_claim_like_sentence(sentence)

        if claim_like and consecutive_claims >= 2:
            lead = _prose_transition(arranged[-1] if arranged else "", sentence)
            core = strip_citations(sentence)
            cits = _sentence_citations(sentence)
            sentence = _format_sentence(f"{lead} {core}", cits)
            consecutive_claims = 1
        else:
            consecutive_claims = consecutive_claims + 1 if claim_like else 0

        arranged.append(sentence)

    arranged = _merge_or_drop_orphan_fragments(arranged)
    arranged_text = dedupe_sentences(" ".join(arranged), threshold=0.70)

    clean_sentences = split_sentences(arranged_text)
    paragraphs: List[str] = []
    current: List[str] = []

    for idx, sentence in enumerate(clean_sentences, start=1):
        current.append(sentence)
        if len(current) >= 4 or idx == len(clean_sentences):
            paragraphs.append(" ".join(current).strip())
            current = []

    rewritten = "\n\n".join(paragraphs).strip()
    rewritten = repair_fragmented_numbers(normalize_punctuation_spacing(rewritten))
    return sanitize_body(rewritten)


def improve_readability(body: str) -> str:
    """Deterministic readability pass: split long clauses and reduce repetition."""
    if not body.strip():
        return body

    body = repair_fragmented_numbers(normalize_punctuation_spacing(body))

    rewritten: List[str] = []
    seen_norm: List[Set[str]] = []

    for sentence in split_sentences(body):
        citations = _sentence_citations(sentence)
        core = strip_citations(sentence)
        if not core:
            continue

        clauses = re.split(r";|:\s+", core, flags=re.IGNORECASE)
        clauses = [clause.strip() for clause in clauses if clause.strip()]
        if not clauses:
            continue

        candidate_sentences: List[str] = []
        for clause in clauses:
            if word_count(clause) > TARGET_AVG_SENTENCE_WORDS:
                candidate_sentences.extend(_split_by_words(clause, TARGET_AVG_SENTENCE_WORDS))
            else:
                candidate_sentences.append(clause)

        for idx, candidate in enumerate(candidate_sentences):
            formatted = _format_sentence(candidate, citations if idx == len(candidate_sentences) - 1 else [])
            if not formatted:
                continue

            candidate_norm = normalize_tokens(strip_citations(formatted))
            if not candidate_norm:
                continue

            is_duplicate = any(jaccard(candidate_norm, prior) >= 0.75 for prior in seen_norm)
            if is_duplicate:
                continue

            seen_norm.append(candidate_norm)
            rewritten.append(formatted)

    rewritten = _merge_or_drop_orphan_fragments(rewritten)
    out = " ".join(rewritten).strip()
    out = dedupe_sentences(out, threshold=0.72)
    out = dedupe_by_claim_and_citation(out)
    out = repair_fragmented_numbers(out)
    out = normalize_punctuation_spacing(out)
    out = sanitize_body(out)
    return out

# -------------------------
# Output quality checks
# -------------------------
def output_quality_gate(
    title: str,
    body: str,
    required_ids: List[int],
    min_words: int,
    max_words: int,
) -> Tuple[bool, Dict]:
    wc = word_count(body)
    rep = repeated_ngram_ratio(body, 4)
    missing_ids = [sid for sid in required_ids if sid not in extract_citation_ids(body)]
    filler = filler_count(body)
    readability = readability_metrics(body)
    avg_sentence_words = readability["avg_sentence_words"]
    max_sentence_words = readability["max_sentence_words"]
    long_sentence_ratio = readability["long_sentence_ratio"]

    ok = (
        min_words <= wc <= max_words and
        rep < 0.32 and
        len(missing_ids) == 0 and
        filler <= 4 and
        avg_sentence_words <= 24 and
        max_sentence_words <= 38 and
        long_sentence_ratio <= 0.12
    )

    return ok, {
        "word_count": wc,
        "repeated_ngram_ratio": rep,
        "missing_ids": missing_ids,
        "filler_count": filler,
        "avg_sentence_words": avg_sentence_words,
        "max_sentence_words": max_sentence_words,
        "long_sentence_ratio": long_sentence_ratio,
    }

# -------------------------
# Main
# -------------------------
def run_agent2(agent1_input_path: str = "agent1_output.json", agent2_output_path: str = "agent2_output.json") -> Dict:
    agent1_data = load_json(agent1_input_path)
    topic, definitions, sources, revision_feedback = parse_agent1(agent1_data)
    required_ids = sorted({s["id"] for s in sources})
    required_ids_set = set(required_ids)
    min_words, max_words, dynamic_min_target = length_policy(sources)

    best_obj = None
    best_score = -10**9
    min_target = dynamic_min_target

    # 1) Draft generation retries
    for _ in range(MAX_RETRIES):
        prompt = build_generation_prompt(
            topic,
            definitions,
            sources,
            min_target,
            max_words,
            revision_feedback=revision_feedback,
        )
        raw = call_ollama(prompt, GEN_TEMPERATURE, GEN_TOP_P, GEN_MAX_TOKENS)
        obj = parse_json_response(raw, stage="generation")

        title = str(obj.get("title", "Technical Analysis")).strip()
        body = sanitize_body(str(obj.get("body", "")).strip(), allowed_ids=required_ids_set)
        body = improve_readability(body)
        body = remap_citations_by_sentence(body, sources, required_ids)
        body = enforce_evidence_alignment(body, sources, required_ids)
        body = article_style_rewrite(body)
        body = improve_readability(body)
        used = obj.get("used_source_ids", [])
        if not isinstance(used, list):
            used = []

        missing = [sid for sid in required_ids if sid not in extract_citation_ids(body)]
        if missing:
            body = inject_missing_citations(body, missing, sources)

        ok, stats = output_quality_gate(title, body, required_ids, min_words=min_words, max_words=max_words)

        score = (
            (1200 if len(stats["missing_ids"]) == 0 else 0) +
            (900 if min_words <= stats["word_count"] <= max_words else 0) -
            abs(int((min_words + max_words) / 2) - stats["word_count"]) -
            int(stats["repeated_ngram_ratio"] * 650) -
            int(stats["filler_count"] * 60) -
            int(max(0.0, stats["avg_sentence_words"] - TARGET_AVG_SENTENCE_WORDS) * 22) -
            int(max(0.0, stats["max_sentence_words"] - MAX_SENTENCE_WORDS) * 6) -
            int(stats["long_sentence_ratio"] * 240)
        )

        current_obj = {
            "title": title,
            "body": body,
            "used_source_ids": required_ids[:]
        }

        if score > best_score:
            best_score = score
            best_obj = current_obj

        if ok:
            break

        if stats["word_count"] < min_words:
            min_target = min(max_words, min_target + 30)

    # 2) Polish pass
    polish_prompt = build_polish_prompt(
        best_obj["title"],
        best_obj["body"],
        required_ids,
        min_words_target=min_words,
        max_words_target=max_words,
        revision_feedback=revision_feedback,
    )
    polished_raw = call_ollama(polish_prompt, POLISH_TEMPERATURE, POLISH_TOP_P, POLISH_MAX_TOKENS)
    polished_obj = parse_json_response(polished_raw, stage="polish")

    p_title = str(polished_obj.get("title", best_obj["title"])).strip()
    p_body = sanitize_body(str(polished_obj.get("body", best_obj["body"])).strip(), allowed_ids=required_ids_set)
    p_body = improve_readability(p_body)
    p_body = remap_citations_by_sentence(p_body, sources, required_ids)
    p_body = enforce_evidence_alignment(p_body, sources, required_ids)
    p_body = article_style_rewrite(p_body)
    p_body = improve_readability(p_body)

    p_missing = [sid for sid in required_ids if sid not in extract_citation_ids(p_body)]
    if p_missing:
        p_body = inject_missing_citations(p_body, p_missing, sources)

    if word_count(p_body) >= min_words:
        final_title, final_body = p_title, p_body
    else:
        final_title, final_body = best_obj["title"], best_obj["body"]

    # 3) Deterministic trim + citation safety
    final_body = deterministic_trim(final_body)
    final_body = improve_readability(final_body)
    final_body = sanitize_body(final_body, allowed_ids=required_ids_set)
    final_body = remap_citations_by_sentence(final_body, sources, required_ids)
    final_body = enforce_evidence_alignment(final_body, sources, required_ids)
    final_body = article_style_rewrite(final_body)
    final_body = improve_readability(final_body)
    final_body = dedupe_by_claim_and_citation(final_body)

    final_missing = [sid for sid in required_ids if sid not in extract_citation_ids(final_body)]
    if final_missing:
        final_body = inject_missing_citations(final_body, final_missing, sources)

    if word_count(final_body) < min_words:
        fallback_body = best_obj["body"]
        final_body = clamp_to_max_words(sanitize_body(fallback_body, allowed_ids=required_ids_set), max_words)
        fallback_missing = [sid for sid in required_ids if sid not in extract_citation_ids(final_body)]
        if fallback_missing:
            final_body = inject_missing_citations(final_body, fallback_missing, sources)

    if word_count(final_body) > max_words:
        final_body = clamp_to_max_words(final_body, max_words)

    used_ids = required_ids[:]

    output = {
        "agent2_output": {
            "title": final_title,
            "body": final_body,
            "used_source_ids": used_ids
        }
    }
    save_json(agent2_output_path, output)
    return output


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "agent1_output.json"
    default_output = script_dir / "agent2_output.json"

    parser = argparse.ArgumentParser(description="Run Agent 2 draft+polish generation against Ollama")
    parser.add_argument(
        "--input",
        dest="input_path",
        default=str(default_input),
        help="Path to agent1_output.json",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        default=str(default_output),
        help="Path to write agent2_output.json",
    )
    parser.add_argument(
        "--model",
        dest="model",
        default=None,
        help="Optional Ollama model name override (same as OLLAMA_MODEL)",
    )
    parser.add_argument(
        "--url",
        dest="ollama_url",
        default=None,
        help="Optional Ollama URL override (same as OLLAMA_URL)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.model:
        OLLAMA_MODEL = args.model
    if args.ollama_url:
        OLLAMA_URL = args.ollama_url

    result = run_agent2(args.input_path, args.output_path)
    print(f"✅ Generated {args.output_path}")
    print(f"Model: {OLLAMA_MODEL}")
    print(f"Ollama URL: {OLLAMA_URL}")