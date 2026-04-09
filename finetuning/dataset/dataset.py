import os
import re
import json
import time
import random
import hashlib
import unicodedata
from typing import Dict, List, Any, Tuple
from collections import deque
from datetime import datetime, timezone

from datasets import load_dataset
import google.generativeai as genai


# =========================================================
# CONFIG
# =========================================================

HF_DATASET = "Sachin21112004/news-tech-dataset"
HF_SPLIT = "train"

# Target new fixed pool
NUM_SAMPLES = 300
SEED = 42

OUTPUT_JSONL = "dataset.jsonl"
REJECTS_JSONL = "rejects.jsonl"

# New 300-index file (generated once, reused forever)
PICKED_INDICES_FILE = "picked_indices_300.json"
# Old 100-index file to avoid overlap
OLD_PICKED_INDICES_FILE = "picked_indices.json"

DONE_HASHES_FILE = "done_hashes.json"

AGENT1_MODEL = "gemini-2.5-flash-lite"
AGENT2_MODEL = "gemini-2.5-flash-lite"

MIN_ARTICLE_WORDS = 400
MIN_SOURCES_PER_AGENT1 = 4
MAX_SOURCES_PER_AGENT1 = 6
MIN_AGENT2_WORDS = 150
MAX_AGENT2_WORDS = 650

GEMINI_MAX_RPM = 6
GEMINI_MAX_RPD = 17

MAX_AGENT1_ATTEMPTS = 1
MAX_AGENT2_ATTEMPTS = 1


# =========================================================
# HELPERS
# =========================================================

def clean_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()

def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))

def strip_fences(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^```(?:json)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    return s

def parse_json(raw: str) -> Dict[str, Any]:
    return json.loads(strip_fences(raw))

def extract_citation_ids(text: str) -> List[int]:
    return sorted(set(int(x) for x in re.findall(r"\[(\d+)\]", text or "")))

def stable_hash(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]

def dedupe_sentences(text: str) -> str:
    sents = re.split(r'(?<=[.!?])\s+', (text or "").strip())
    kept, seen = [], set()
    for s in sents:
        t = s.strip()
        if not t:
            continue
        k = re.sub(r"\W+", "", t.lower())
        if len(k) > 20 and k in seen:
            continue
        seen.add(k)
        kept.append(t)
    return " ".join(kept).strip()

def fix_mojibake(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "â€™": "'", "â€˜": "'", "â€œ": '"', "â€": '"',
        "â€“": "-", "â€”": "-", "â€¦": "...", "Â": " ", "Ã—": "x", "�": "",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
    return clean_ws(text)

def load_json_file(path: str, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def save_json_file(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =========================================================
# STRICT ARTICLE FILTER
# =========================================================

CORE_STRONG = [
    "benchmark", "latency", "throughput", "architecture", "kernel", "compiler",
    "api", "sdk", "protocol", "encryption", "cve", "vulnerability", "patch",
    "gpu", "cpu", "npu", "semiconductor", "foundry", "datacenter", "inference",
    "training", "quantization", "transformer", "llm", "firmware", "microarchitecture"
]
CORE_MED = [
    "machine learning", "artificial intelligence", "cloud", "model", "neural",
    "robotics", "5g", "chipset", "battery chemistry"
]
NEG_BAD = [
    "coupon", "deal", "sale", "best price", "opening weekend", "box office",
    "gift guide", "startup battlefield", "influencer", "celebrity", "promo"
]

def is_core_technical(title: str, content: str) -> bool:
    t = f"{title} {content}".lower()
    if any(k in t for k in NEG_BAD):
        return False
    strong = sum(1 for k in CORE_STRONG if k in t)
    med = sum(1 for k in CORE_MED if k in t)
    score = strong * 3 + med
    return strong >= 1 and score >= 4


# =========================================================
# QUOTA LIMITER
# =========================================================

class GeminiQuotaLimiter:
    def __init__(self, max_rpm: int, max_rpd: int, jitter_sec: float = 0.8):
        self.max_rpm = max_rpm
        self.max_rpd = max_rpd
        self.jitter_sec = jitter_sec
        self.minute_window = deque()
        self.day_count = 0
        self.day_key = self._utc_day_key()

    def _utc_day_key(self):
        now = datetime.now(timezone.utc)
        return (now.year, now.month, now.day)

    def _roll_day_if_needed(self):
        k = self._utc_day_key()
        if k != self.day_key:
            self.day_key = k
            self.day_count = 0

    def _prune_window(self, now_ts: float):
        while self.minute_window and (now_ts - self.minute_window[0] >= 60.0):
            self.minute_window.popleft()

    def before_request(self):
        self._roll_day_if_needed()
        if self.day_count >= self.max_rpd:
            raise RuntimeError(f"Gemini RPD reached: {self.day_count}/{self.max_rpd}. Stop now.")
        while True:
            now = time.time()
            self._prune_window(now)
            if len(self.minute_window) < self.max_rpm:
                break
            wait_for = 60.0 - (now - self.minute_window[0])
            if wait_for > 0:
                time.sleep(wait_for + random.uniform(0, self.jitter_sec))

    def mark_success(self):
        now = time.time()
        self._roll_day_if_needed()
        self._prune_window(now)
        self.minute_window.append(now)
        self.day_count += 1

    def stats(self):
        self._roll_day_if_needed()
        return {
            "rpm_used_current_window": len(self.minute_window),
            "rpm_limit": self.max_rpm,
            "rpd_used_today": self.day_count,
            "rpd_limit": self.max_rpd,
            "utc_day": self.day_key,
        }


# =========================================================
# GEMINI CLIENT
# =========================================================

def configure_gemini():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise EnvironmentError("GEMINI_API_KEY not set")
    genai.configure(api_key=key)

def get_gemini_model(model_name: str):
    return genai.GenerativeModel(
        model_name,
        system_instruction="Return strict JSON only. Use only provided evidence.",
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0.2,
            "max_output_tokens": 1200
        }
    )

def call_gemini_json(model, limiter: GeminiQuotaLimiter, prompt: str) -> str:
    limiter.before_request()
    resp = model.generate_content(prompt)
    limiter.mark_success()
    return (resp.text or "").strip()


# =========================================================
# AGENTS (BOTH GEMINI)
# =========================================================

def build_agent1_with_gemini(agent1_model, limiter: GeminiQuotaLimiter, title: str, url: str, article_text: str) -> Dict[str, Any]:
    clipped = article_text[:14000]
    prompt = f"""
Extract structured research data from this technical article.

URL: {url}
TITLE: {title}
TEXT:
{clipped}

Return VALID JSON ONLY:
{{
  "research_data": {{
    "definitions": {{
      "term": "definition"
    }},
    "sources": [
      {{
        "id": 1,
        "subtopic": "specific subtopic",
        "extracted_claim": "single concrete factual claim directly supported by raw_chunk",
        "raw_chunk": "supporting chunk from article",
        "url": "{url}"
      }}
    ]
  }}
}}

Rules:
- definitions: 2 to 5
- sources: {MIN_SOURCES_PER_AGENT1} to {MAX_SOURCES_PER_AGENT1}
- ids sequential starting at 1
- no hallucinations
- no extra keys
"""
    raw = call_gemini_json(agent1_model, limiter, prompt)
    data = parse_json(raw)

    rd = data.get("research_data", {})
    defs = rd.get("definitions", {})
    srcs = rd.get("sources", [])

    if not isinstance(defs, dict):
        defs = {}
    if not isinstance(srcs, list):
        srcs = []

    srcs = srcs[:MAX_SOURCES_PER_AGENT1]
    norm = []
    for i, s in enumerate(srcs, start=1):
        if not isinstance(s, dict):
            continue
        norm.append({
            "id": i,
            "subtopic": clean_ws(str(s.get("subtopic", ""))),
            "extracted_claim": clean_ws(str(s.get("extracted_claim", ""))),
            "raw_chunk": clean_ws(str(s.get("raw_chunk", ""))),
            "url": url
        })

    return {"research_data": {"definitions": defs, "sources": norm}}

def build_agent2_with_gemini(agent2_model, limiter: GeminiQuotaLimiter, agent1: Dict[str, Any], min_words_target: int = 220) -> Dict[str, Any]:
    rd = agent1["research_data"]
    defs = rd.get("definitions", {})
    srcs = rd.get("sources", [])
    req_ids = [s["id"] for s in srcs]

    defs_text = "\n".join([f"- {k}: {v}" for k, v in defs.items()]) if defs else "- None"
    src_blocks = []
    for s in srcs:
        src_blocks.append(
            f"[{s['id']}]\nSubtopic: {s['subtopic']}\nClaim: {s['extracted_claim']}\nEvidence: {s['raw_chunk']}\nURL: {s['url']}\n"
        )

    prompt = f"""
Write concise technical analysis grounded ONLY in provided evidence.

DEFINITIONS:
{defs_text}

SOURCES:
{chr(10).join(src_blocks)}

Requirements:
- Target length: {min_words_target}-600 words
- Inline citations only [1], [2], ...
- Must use ALL source ids: {req_ids}
- No hallucinations
- No URL list
- No repeated sentences

Return JSON:
{{
  "agent2_output": {{
    "title": "string",
    "body": "string",
    "used_source_ids": [1,2,3]
  }}
}}
"""
    raw = call_gemini_json(agent2_model, limiter, prompt)
    data = parse_json(raw)
    a2 = data.get("agent2_output", {})

    title = clean_ws(str(a2.get("title", "Technical Analysis")))
    body = dedupe_sentences(fix_mojibake(clean_ws(str(a2.get("body", "")))))
    used = []
    for x in a2.get("used_source_ids", []):
        try:
            used.append(int(x))
        except Exception:
            pass
    used = sorted(set(used) | set(req_ids))

    cited = set(extract_citation_ids(body))
    miss = [i for i in req_ids if i not in cited]
    if miss:
        body += "\n\n" + " ".join([f"Supporting evidence [{i}]." for i in miss])

    return {"agent2_output": {"title": title, "body": body, "used_source_ids": used}}


# =========================================================
# VALIDATION
# =========================================================

def validate_agent1(agent1: Dict[str, Any]) -> Tuple[bool, str]:
    rd = agent1.get("research_data")
    if not isinstance(rd, dict):
        return False, "missing_research_data"
    defs = rd.get("definitions")
    srcs = rd.get("sources")
    if not isinstance(defs, dict):
        return False, "definitions_invalid"
    if not isinstance(srcs, list) or len(srcs) < MIN_SOURCES_PER_AGENT1:
        return False, "few_sources"

    for i, s in enumerate(srcs, start=1):
        if not isinstance(s, dict):
            return False, "source_not_object"
        for k in ["id", "subtopic", "extracted_claim", "raw_chunk", "url"]:
            if k not in s:
                return False, f"missing_{k}"
        if s["id"] != i:
            return False, "id_sequence_error"
    return True, "ok"

def validate_agent2(agent1: Dict[str, Any], agent2: Dict[str, Any]) -> Tuple[bool, str]:
    a2 = agent2.get("agent2_output")
    if not isinstance(a2, dict):
        return False, "missing_agent2_output"
    if not a2.get("title") or not a2.get("body"):
        return False, "empty_title_or_body"

    wc = word_count(a2["body"])
    if wc < MIN_AGENT2_WORDS or wc > MAX_AGENT2_WORDS:
        return False, f"body_wordcount_out_of_range:{wc}"

    for c in re.findall(r"\[([^\]]+)\]", a2["body"]):
        if not c.strip().isdigit():
            return False, f"non_numeric_citation:[{c}]"

    req_ids = sorted([s["id"] for s in agent1["research_data"]["sources"]])
    used_ids = sorted(set(int(x) for x in a2["used_source_ids"]))
    if used_ids != req_ids:
        return False, "used_source_ids_mismatch"

    cited = set(extract_citation_ids(a2["body"]))
    missing = [i for i in req_ids if i not in cited]
    if missing:
        return False, f"missing_citations:{missing}"

    return True, "ok"

def to_jsonl_row(agent1: Dict[str, Any], agent2: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": json.dumps(agent1, ensure_ascii=False)},
            {"role": "assistant", "content": json.dumps(agent2, ensure_ascii=False)},
        ]
    }


# =========================================================
# FIXED INDEX SAMPLING (NEW 300, EXCLUDING OLD 100)
# =========================================================

def build_or_load_fixed_indices(rows: List[Dict[str, Any]]) -> List[int]:
    # Reuse if already generated
    saved = load_json_file(PICKED_INDICES_FILE, default=None)
    if saved and isinstance(saved, list) and len(saved) > 0:
        return saved[:NUM_SAMPLES]

    old_picked = set(load_json_file(OLD_PICKED_INDICES_FILE, default=[]))
    print(f"Excluding old picked indices from {OLD_PICKED_INDICES_FILE}: {len(old_picked)}")

    valid_indices = []
    for i, r in enumerate(rows):
        if i in old_picked:
            continue
        title = fix_mojibake(clean_ws(str(r.get("title", ""))))
        content = fix_mojibake(clean_ws(str(r.get("content", ""))))
        if is_core_technical(title, content):
            valid_indices.append(i)

    rnd = random.Random(SEED)
    rnd.shuffle(valid_indices)

    if len(valid_indices) < NUM_SAMPLES:
        print(f"[warn] only {len(valid_indices)} eligible new indices found (requested {NUM_SAMPLES})")

    picked = valid_indices[:NUM_SAMPLES]
    save_json_file(PICKED_INDICES_FILE, picked)
    return picked

def load_articles_fixed() -> List[Dict[str, str]]:
    print("Downloading Hugging Face dataset...")
    ds = load_dataset(HF_DATASET, split=HF_SPLIT)

    print("Primary filter: English + category tech + length")
    filtered = ds.filter(
        lambda x: (x.get("language") and str(x["language"]).lower().startswith("en")) and
                  (x.get("category") and "tech" in str(x["category"]).lower()) and
                  (x.get("content") and len(str(x["content"]).split()) >= MIN_ARTICLE_WORDS)
    )
    print(f"Primary filtered rows: {filtered.num_rows}")

    rows = [r for r in filtered]
    picked_indices = build_or_load_fixed_indices(rows)
    print(f"Using fixed indices from {PICKED_INDICES_FILE}: {len(picked_indices)}")

    out = []
    for idx in picked_indices:
        r = rows[idx]
        out.append({
            "title": fix_mojibake(clean_ws(str(r.get("title", "")))) or f"Tech Article {idx}",
            "url": clean_ws(str(r.get("url", ""))) or f"hf://{HF_DATASET}/{HF_SPLIT}/{idx}",
            "content": fix_mojibake(clean_ws(str(r.get("content", ""))))
        })
    return out


# =========================================================
# MAIN
# =========================================================

def main():
    configure_gemini()
    agent1_model = get_gemini_model(AGENT1_MODEL)
    agent2_model = get_gemini_model(AGENT2_MODEL)
    limiter = GeminiQuotaLimiter(max_rpm=GEMINI_MAX_RPM, max_rpd=GEMINI_MAX_RPD)

    articles = load_articles_fixed()
    print(f"\nSelected strict core-tech articles: {len(articles)}")
    print(f"Gemini guard => RPM: {GEMINI_MAX_RPM}, RPD: {GEMINI_MAX_RPD}")

    done_hashes = set(load_json_file(DONE_HASHES_FILE, default=[]))
    kept, skipped = 0, 0

    with open(OUTPUT_JSONL, "a", encoding="utf-8") as out_f, \
         open(REJECTS_JSONL, "a", encoding="utf-8") as rej_f:

        for i, art in enumerate(articles, start=1):
            title, url, content = art["title"], art["url"], art["content"]
            print(f"\n[{i}/{len(articles)}] {title[:100]}")

            h = stable_hash(content[:12000])
            if h in done_hashes:
                skipped += 1
                print("  - skipped (already processed hash)")
                continue

            try:
                agent1 = build_agent1_with_gemini(agent1_model, limiter, title, url, content)
                v1, m1 = validate_agent1(agent1)
                if not v1:
                    skipped += 1
                    rej_f.write(json.dumps({"idx": i, "url": url, "reason": f"agent1_invalid:{m1}"}, ensure_ascii=False) + "\n")
                    done_hashes.add(h)
                    save_json_file(DONE_HASHES_FILE, sorted(done_hashes))
                    continue
            except RuntimeError as qe:
                print(f"  - hard quota stop: {qe}")
                save_json_file(DONE_HASHES_FILE, sorted(done_hashes))
                print("Stopping safely.")
                return
            except Exception as e:
                skipped += 1
                rej_f.write(json.dumps({"idx": i, "url": url, "reason": f"agent1_failed:{str(e)}"}, ensure_ascii=False) + "\n")
                done_hashes.add(h)
                save_json_file(DONE_HASHES_FILE, sorted(done_hashes))
                continue

            try:
                agent2 = build_agent2_with_gemini(agent2_model, limiter, agent1, min_words_target=220)
                v2, m2 = validate_agent2(agent1, agent2)
                if not v2:
                    skipped += 1
                    rej_f.write(json.dumps({"idx": i, "url": url, "reason": f"agent2_invalid:{m2}"}, ensure_ascii=False) + "\n")
                    done_hashes.add(h)
                    save_json_file(DONE_HASHES_FILE, sorted(done_hashes))
                    continue
            except RuntimeError as qe:
                print(f"  - hard quota stop: {qe}")
                save_json_file(DONE_HASHES_FILE, sorted(done_hashes))
                print("Stopping safely.")
                return
            except Exception as e:
                skipped += 1
                rej_f.write(json.dumps({"idx": i, "url": url, "reason": f"agent2_failed:{str(e)}"}, ensure_ascii=False) + "\n")
                done_hashes.add(h)
                save_json_file(DONE_HASHES_FILE, sorted(done_hashes))
                continue

            out_f.write(json.dumps(to_jsonl_row(agent1, agent2), ensure_ascii=False) + "\n")
            kept += 1
            done_hashes.add(h)
            save_json_file(DONE_HASHES_FILE, sorted(done_hashes))
            print(f"  + kept ({kept}) | quota: {limiter.stats()}")

    print("\n========================")
    print(f"Done. Kept={kept}, Skipped={skipped}")
    print(f"Saved: {OUTPUT_JSONL}")
    print(f"Rejects: {REJECTS_JSONL}")
    print(f"Fixed indices: {PICKED_INDICES_FILE}")
    print("========================")


if __name__ == "__main__":
    main()