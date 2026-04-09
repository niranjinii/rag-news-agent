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
GEN_MAX_TOKENS = 2400

POLISH_TEMPERATURE = 0.08
POLISH_TOP_P = 0.9
POLISH_MAX_TOKENS = 1800

MIN_WORDS = 700
MAX_WORDS = 1000

TARGET_MIN_GEN_WORDS = 780
MAX_RETRIES = 3

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

def sanitize_body(text: str) -> str:
    text = re.sub(r"(?im)^\s*(references|bibliography|sources)\s*:?\s*$", "", text)
    text = re.sub(r"https?://\S+", "", text)

    # remove leaked metadata line inside body
    text = re.sub(r"(?im)^\s*used[_\s]*source[_\s]*ids\s*:\s*\[[^\]]*\]\s*$", "", text)

    # optional: remove ugly citation clusters like [4] [2]
    text = re.sub(r"(?:\[\d+\]\s*){2,}", "", text)

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

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
def parse_agent1(agent1_data: Dict) -> Tuple[str, Dict[str, str], List[Dict]]:
    research_data = agent1_data.get("research_data", {})
    topic = research_data.get("topic", "Technical Topic")
    definitions = research_data.get("definitions", {})
    sources = research_data.get("sources", [])

    if not sources:
        raise ValueError("No sources found in research_data.sources")

    seen_ids = set()
    for s in sources:
        if "id" not in s:
            raise ValueError("Each source must contain 'id'")
        sid = int(s["id"])
        if sid in seen_ids:
            raise ValueError(f"Duplicate source id found: {sid}")
        seen_ids.add(sid)
        s["id"] = sid
    return topic, definitions, sources

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
def build_generation_prompt(topic: str, definitions: Dict[str, str], sources: List[Dict], min_words_target: int) -> str:
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

    required_ids = [s["id"] for s in sources]
    required_ids_str = ", ".join([str(i) for i in required_ids])

    return f"""
You are a senior technical analyst. Produce one grounded technical article from the provided evidence only.

TOPIC:
{topic}

DEFINITIONS:
{defs_text}

EVIDENCE:
{chr(10).join(blocks)}

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

ANALYSIS RULES (MANDATORY):
- If the topic/evidence includes multiple products, provide clear, evidence-backed comparison points.
- If the topic/evidence is a single product, focus on architecture/specs, measured performance, and practical usage implications.
- Include at least 2 short "what this means in practice" statements tied to cited facts.
- Include one concise recommendation statement based only on cited evidence.

LENGTH:
- body must be {min_words_target}-{MAX_WORDS} words.

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

def build_polish_prompt(title: str, body: str, source_ids: List[int]) -> str:
    ids = ", ".join([str(i) for i in source_ids])
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
6) Body length: {MIN_WORDS}-{MAX_WORDS} words.

Return ONLY valid JSON with same 3 keys.
Escape newlines as \\n inside JSON strings.
""".strip()

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

# -------------------------
# Output quality checks
# -------------------------
def output_quality_gate(title: str, body: str, required_ids: List[int]) -> Tuple[bool, Dict]:
    wc = word_count(body)
    rep = repeated_ngram_ratio(body, 4)
    missing_ids = [sid for sid in required_ids if sid not in extract_citation_ids(body)]
    filler = filler_count(body)

    ok = (
        MIN_WORDS <= wc <= MAX_WORDS and
        rep < 0.32 and
        len(missing_ids) == 0 and
        filler <= 4
    )

    return ok, {
        "word_count": wc,
        "repeated_ngram_ratio": rep,
        "missing_ids": missing_ids,
        "filler_count": filler,
    }

# -------------------------
# Main
# -------------------------
def run_agent2(agent1_input_path: str = "agent1_output.json", agent2_output_path: str = "agent2_output.json") -> Dict:
    agent1_data = load_json(agent1_input_path)
    topic, definitions, sources = parse_agent1(agent1_data)
    required_ids = sorted([s["id"] for s in sources])

    best_obj = None
    best_score = -10**9
    min_target = TARGET_MIN_GEN_WORDS

    # 1) Draft generation retries
    for _ in range(MAX_RETRIES):
        prompt = build_generation_prompt(topic, definitions, sources, min_target)
        raw = call_ollama(prompt, GEN_TEMPERATURE, GEN_TOP_P, GEN_MAX_TOKENS)
        obj = parse_json_response(raw, stage="generation")

        title = str(obj.get("title", "Technical Analysis")).strip()
        body = sanitize_body(str(obj.get("body", "")).strip())
        used = obj.get("used_source_ids", [])
        if not isinstance(used, list):
            used = []

        missing = [sid for sid in required_ids if sid not in extract_citation_ids(body)]
        if missing:
            body = inject_missing_citations(body, missing, sources)

        ok, stats = output_quality_gate(title, body, required_ids)

        score = (
            (1200 if len(stats["missing_ids"]) == 0 else 0) +
            (900 if MIN_WORDS <= stats["word_count"] <= MAX_WORDS else 0) -
            abs(850 - stats["word_count"]) -
            int(stats["repeated_ngram_ratio"] * 650) -
            int(stats["filler_count"] * 60)
        )

        current_obj = {
            "title": title,
            "body": body,
            "used_source_ids": sorted(set([int(x) for x in used if str(x).isdigit()] + required_ids))
        }

        if score > best_score:
            best_score = score
            best_obj = current_obj

        if ok:
            break

        if stats["word_count"] < MIN_WORDS:
            min_target = min(MAX_WORDS, min_target + 40)

    # 2) Polish pass
    polish_prompt = build_polish_prompt(best_obj["title"], best_obj["body"], required_ids)
    polished_raw = call_ollama(polish_prompt, POLISH_TEMPERATURE, POLISH_TOP_P, POLISH_MAX_TOKENS)
    polished_obj = parse_json_response(polished_raw, stage="polish")

    p_title = str(polished_obj.get("title", best_obj["title"])).strip()
    p_body = sanitize_body(str(polished_obj.get("body", best_obj["body"])).strip())

    p_missing = [sid for sid in required_ids if sid not in extract_citation_ids(p_body)]
    if p_missing:
        p_body = inject_missing_citations(p_body, p_missing, sources)

    if word_count(p_body) >= 680:
        final_title, final_body = p_title, p_body
    else:
        final_title, final_body = best_obj["title"], best_obj["body"]

    # 3) Deterministic trim + citation safety
    final_body = deterministic_trim(final_body)

    final_missing = [sid for sid in required_ids if sid not in extract_citation_ids(final_body)]
    if final_missing:
        final_body = inject_missing_citations(final_body, final_missing, sources)

    if word_count(final_body) < MIN_WORDS:
        fallback_body = best_obj["body"]
        final_body = clamp_to_max_words(sanitize_body(fallback_body), MAX_WORDS)
        fallback_missing = [sid for sid in required_ids if sid not in extract_citation_ids(final_body)]
        if fallback_missing:
            final_body = inject_missing_citations(final_body, fallback_missing, sources)

    used_ids = sorted(list(extract_citation_ids(final_body).intersection(set(required_ids))))
    if len(used_ids) < len(required_ids):
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