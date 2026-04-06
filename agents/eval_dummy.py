"""
Agent3: Editor/Evaluator implementation.

Implements:
1) Factual Accuracy (95% pass)
   - Layer 1 (Regex / 25%)
   - Layer 2 (Vector Similarity / 60%)
   - Layer 3 (LLM Disambiguation / 15%)

2) Citation Quality (90% pass)

Decision:
- REWRITE_ACCURACY (hard) when accuracy < 95 or citation quality < 90
- PUBLISH otherwise

This file also exposes `evaluation_agent_node` to keep compatibility with the
existing graph pipeline contract.
"""

import os
import re
import json
from functools import lru_cache
from datetime import datetime
from typing import Any

import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from state import PipelineState, Evaluation


class EditorAgent:
    """Evaluates factual accuracy and citation quality for a draft article."""

    def __init__(self, groq_api_key: str | None):
        """Initialize LLM + embedding dependencies.

        Args:
            groq_api_key: Groq API key. If absent, LLM disambiguation returns 0.
        """
        self.groq_api_key = groq_api_key
        self.llm = Groq(api_key=groq_api_key) if groq_api_key else None
        configured_model = os.getenv("GROQ_MODEL", "").strip()
        base_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        self.llm_models = [configured_model, *base_models] if configured_model else base_models
        deduped_models: list[str] = []
        for model_name in self.llm_models:
            if model_name and model_name not in deduped_models:
                deduped_models.append(model_name)
        self.llm_models = deduped_models
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.trace: list[dict[str, Any]] = []

    def _log(self, stage: str, message: str, **data: Any) -> None:
        """Store structured trace logs for JSON debug output."""
        self.trace.append(
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "stage": stage,
                "message": message,
                "data": data,
            }
        )

    def evaluate(self, draft: str, context_data: dict) -> dict:
        """Run full evaluation and return decision payload."""
        self.trace = []
        self._log(
            "evaluate",
            "Starting evaluation",
            draft_chars=len(draft or ""),
            source_count=len(context_data.get("sources", [])) if isinstance(context_data, dict) else 0,
        )
        accuracy = self._check_accuracy(draft, context_data)
        citation = self._verify_citation_relevance(draft, context_data)
        result = self._make_decision(accuracy, citation)
        result["trace"] = self.trace
        self._log(
            "evaluate",
            "Completed evaluation",
            decision=result.get("decision"),
            accuracy_score=accuracy.get("score"),
            citation_score=citation.get("citation_score"),
        )
        result["trace"] = self.trace
        return result

    def _check_accuracy(self, draft: str, context: dict) -> dict:
        """Compute hybrid factual accuracy score (0-100)."""
        self._log("accuracy", "Computing 3-layer accuracy")
        layer1 = self._validate_citations_and_specs(draft, context)
        layer2 = self._verify_with_vectors(draft, context)
        layer3 = self._verify_ambiguous_with_llm(layer2["ambiguous_claims"], context)

        layer1_component = (100.0 if layer1["valid"] else 0.0) * 0.25
        layer2_component = layer2["verified_percentage"] * 0.60
        layer3_component = layer3["boost_score"] * 0.15
        score = round(layer1_component + layer2_component + layer3_component, 2)
        self._log(
            "accuracy",
            "Computed weighted accuracy",
            layer1_component=layer1_component,
            layer2_component=layer2_component,
            layer3_component=layer3_component,
            final_score=score,
        )

        return {
            "score": score,
            "layer1": layer1,
            "layer2": layer2,
            "layer3": layer3,
        }

    def _validate_citations_and_specs(self, draft: str, context: dict) -> dict:
        """Layer 1 regex checks for citation integrity and numeric spec grounding."""
        sources = context.get("sources", []) if isinstance(context, dict) else []
        source_ids = {
            src.get("id")
            for src in sources
            if isinstance(src, dict) and isinstance(src.get("id"), int)
        }

        broken_citations: list[str] = []
        all_bracket_tokens = re.findall(r"\[([^\]]+)\]", draft)
        numeric_citations = re.findall(r"\[(\d+)\]", draft)

        for token in all_bracket_tokens:
            if not token.strip().isdigit():
                broken_citations.append(f"Invalid citation token [{token}]")

        for token in numeric_citations:
            source_id = int(token)
            if source_id not in source_ids:
                broken_citations.append(f"Invalid citation [{source_id}]")

        draft_specs = self._extract_specs(draft)
        source_text = "\n".join(self._source_texts(context))
        source_specs = self._extract_specs(source_text)

        hallucinated_specs: list[str] = []
        missing = sorted(draft_specs["general_metrics"] - source_specs["general_metrics"])
        hallucinated_specs.extend([f"Hallucinated metric: {value}" for value in missing])

        valid = len(broken_citations) == 0 and len(hallucinated_specs) == 0
        self._log(
            "layer1_regex",
            "Validated citations and specs",
            valid=valid,
            total_bracket_tokens=len(all_bracket_tokens),
            total_numeric_citations=len(numeric_citations),
            broken_citation_count=len(broken_citations),
            hallucinated_spec_count=len(hallucinated_specs),
        )
        return {
            "valid": valid,
            "broken_citations": broken_citations,
            "hallucinated_specs": hallucinated_specs,
            "draft_specs": {
                "general_metrics": sorted(draft_specs["general_metrics"]),
            },
        }

    def _verify_with_vectors(self, draft: str, context: dict) -> dict:
        """Layer 2 semantic grounding using sentence embeddings + cosine similarity."""
        draft_sentences = [
            sentence for sentence in self._split_sentences(draft) if self._word_count(sentence) >= 15
        ]
        source_sentences = [
            sentence
            for src_text in self._source_texts(context)
            for sentence in self._split_sentences(src_text)
            if sentence.strip()
        ]

        if not draft_sentences or not source_sentences:
            self._log(
                "layer2_vectors",
                "Skipped vector verification due to empty sentence inputs",
                draft_sentence_count=len(draft_sentences),
                source_sentence_count=len(source_sentences),
            )
            return {
                "verified_percentage": 0.0,
                "hallucinations": [],
                "ambiguous_claims": [],
                "verified_claims": [],
            }

        draft_emb = self.embedding_model.encode(draft_sentences, convert_to_numpy=True)
        source_emb = self.embedding_model.encode(source_sentences, convert_to_numpy=True)
        sims = cosine_similarity(draft_emb, source_emb)

        verified_claims: list[dict[str, Any]] = []
        ambiguous_claims: list[dict[str, Any]] = []
        hallucinations: list[dict[str, Any]] = []

        for idx, sentence in enumerate(draft_sentences):
            best_sim = float(np.max(sims[idx]))
            if best_sim >= 0.75:
                verified_claims.append({"text": sentence, "similarity": round(best_sim, 4)})
            elif 0.65 <= best_sim < 0.75:
                ambiguous_claims.append({"text": sentence, "similarity": round(best_sim, 4)})
            else:
                hallucinations.append({"text": sentence, "similarity": round(best_sim, 4)})

        verified_percentage = round((len(verified_claims) / len(draft_sentences)) * 100.0, 2)
        self._log(
            "layer2_vectors",
            "Completed vector verification",
            draft_sentence_count=len(draft_sentences),
            source_sentence_count=len(source_sentences),
            verified_count=len(verified_claims),
            ambiguous_count=len(ambiguous_claims),
            hallucination_count=len(hallucinations),
            verified_percentage=verified_percentage,
        )
        return {
            "verified_percentage": verified_percentage,
            "hallucinations": hallucinations,
            "ambiguous_claims": ambiguous_claims,
            "verified_claims": verified_claims,
        }

    def _verify_ambiguous_with_llm(self, claims: list[dict], context: dict) -> dict:
        """Layer 3 LLM binary validation for ambiguous claims.

        Returns boost score in [0, 100].
        """
        if not claims:
            self._log("layer3_llm", "No ambiguous claims; full boost applied", boost_score=100.0)
            return {"boost_score": 100.0, "verified": [], "rejected": [], "skipped": []}

        if not self.llm:
            self._log("layer3_llm", "Skipped LLM verification (missing API key)", claim_count=len(claims))
            return {
                "boost_score": 0.0,
                "verified": [],
                "rejected": [],
                "skipped": ["GROQ_API_KEY not set; skipped ambiguous claim validation"],
            }

        verified: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        skipped: list[str] = []

        for claim in claims:
            claim_text = claim.get("text", "")
            if not claim_text:
                continue

            try:
                top_sources = self._select_relevant_sources(claim_text, context, limit=3)
                source_blob = "\n\n".join(
                    [f"Source {idx + 1}: {snippet}" for idx, snippet in enumerate(top_sources)]
                )
                prompt = (
                    "You are a strict fact verifier.\n"
                    "Check if this claim appears in the provided sources.\n"
                    "Answer only YES or NO.\n\n"
                    f"Claim: {claim_text}\n\n"
                    f"Sources:\n{source_blob}"
                )

                last_error = None
                resolved = False
                for model_name in self.llm_models:
                    try:
                        completion = self.llm.chat.completions.create(
                            model=model_name,
                            temperature=0.0,
                            messages=[{"role": "user", "content": prompt}],
                        )
                        answer = (completion.choices[0].message.content or "").strip().upper()

                        claim_with_model = {
                            **claim,
                            "model": model_name,
                            "answer": answer,
                        }

                        if answer.startswith("YES"):
                            verified.append(claim_with_model)
                        else:
                            rejected.append(claim_with_model)
                        resolved = True
                        break
                    except Exception as model_error:  # pragma: no cover - external dependency path
                        last_error = model_error

                if not resolved:
                    skipped.append(
                        f"LLM error for claim: {claim_text[:80]}... ({last_error})"
                    )
            except Exception as error:  # pragma: no cover - external dependency path
                skipped.append(f"LLM error for claim: {claim_text[:80]}... ({error})")

        boost_score = round((len(verified) / len(claims)) * 100.0, 2) if claims else 100.0
        self._log(
            "layer3_llm",
            "Completed ambiguous claim verification",
            models_tried=self.llm_models,
            ambiguous_claim_count=len(claims),
            verified_count=len(verified),
            rejected_count=len(rejected),
            skipped_count=len(skipped),
            boost_score=boost_score,
        )
        return {
            "boost_score": boost_score,
            "verified": verified,
            "rejected": rejected,
            "skipped": skipped,
        }

    def _verify_citation_relevance(self, draft: str, context: dict) -> dict:
        """Check sentence-to-citation semantic relevance (0-100 score)."""
        citation_instances = self._extract_citation_instances(draft)
        sources = context.get("sources", []) if isinstance(context, dict) else []
        source_map = {
            src.get("id"): self._build_source_text(src)
            for src in sources
            if isinstance(src, dict) and isinstance(src.get("id"), int)
        }

        if not citation_instances:
            self._log("citation_quality", "No citations found in draft", citation_score=0.0)
            return {
                "citation_score": 0.0,
                "mis_citations": ["No citations found in draft"],
                "checked": 0,
            }

        source_ids = list(source_map.keys())
        source_texts = [source_map[src_id] for src_id in source_ids]
        source_emb = self.embedding_model.encode(source_texts, convert_to_numpy=True) if source_texts else np.array([])

        passing = 0
        mis_citations: list[str] = []

        for citation_id, sentence in citation_instances:
            if citation_id not in source_map:
                mis_citations.append(f"❌ Invalid citation [{citation_id}] in sentence '{sentence[:120]}'")
                continue

            sent_emb = self.embedding_model.encode([sentence], convert_to_numpy=True)
            target_emb = self.embedding_model.encode([source_map[citation_id]], convert_to_numpy=True)
            direct_sim = float(cosine_similarity(sent_emb, target_emb)[0][0])

            if direct_sim >= 0.70:
                passing += 1
                continue

            suggested = None
            if source_texts:
                sims = cosine_similarity(sent_emb, source_emb)[0]
                best_idx = int(np.argmax(sims))
                suggested = source_ids[best_idx]

            if suggested and suggested != citation_id:
                mis_citations.append(
                    f"⚠️ Sentence '{sentence[:120]}' cites wrong source → should be [{suggested}]"
                )
            else:
                mis_citations.append(
                    f"⚠️ Sentence '{sentence[:120]}' weakly supported by citation [{citation_id}] (sim={direct_sim:.2f})"
                )

        citation_score = round((passing / len(citation_instances)) * 100.0, 2)
        self._log(
            "citation_quality",
            "Completed citation relevance verification",
            checked=len(citation_instances),
            passing=passing,
            failing=len(citation_instances) - passing,
            citation_score=citation_score,
            mis_citation_count=len(mis_citations),
        )
        return {
            "citation_score": citation_score,
            "mis_citations": mis_citations,
            "checked": len(citation_instances),
        }

    def _make_decision(self, accuracy: dict, citation: dict) -> dict:
        """Apply hard-gate decision logic and produce editor feedback."""
        accuracy_score = float(accuracy["score"])
        citation_score = float(citation["citation_score"])

        accuracy_feedback: list[str] = []
        for error in accuracy["layer1"].get("broken_citations", []):
            accuracy_feedback.append(f"❌ {error}")
        for error in accuracy["layer1"].get("hallucinated_specs", []):
            accuracy_feedback.append(f"❌ {error}")
        for claim in accuracy["layer2"].get("ambiguous_claims", []):
            accuracy_feedback.append(
                f"❌ Unverified claim ({claim['similarity']:.2f}): '{claim['text'][:140]}'"
            )
        for claim in accuracy["layer2"].get("hallucinations", []):
            accuracy_feedback.append(
                f"❌ Hallucinated claim ({claim['similarity']:.2f}): '{claim['text'][:140]}'"
            )

        citation_feedback = citation.get("mis_citations", [])

        if accuracy_score < 95.0:
            decision = "REWRITE_ACCURACY"
            hard_reason = f"Accuracy below threshold: {accuracy_score:.2f} < 95.00"
        elif citation_score < 90.0:
            decision = "REWRITE_ACCURACY"
            hard_reason = f"Citation quality below threshold: {citation_score:.2f} < 90.00"
        else:
            decision = "PUBLISH"
            hard_reason = "All thresholds met"

        self._log(
            "decision",
            "Applied hard-gate decision",
            decision=decision,
            reason=hard_reason,
            accuracy_score=accuracy_score,
            citation_score=citation_score,
        )

        return {
            "decision": decision,
            "hard_reason": hard_reason,
            "accuracy": accuracy,
            "citation": citation,
            "accuracy_feedback": accuracy_feedback,
            "citation_feedback": citation_feedback,
        }

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentence-like units."""
        normalized = re.sub(r"\s+", " ", text or "").strip()
        if not normalized:
            return []
        # Added (?=[A-Z]) to prevent splitting decimals or acronyms
        return [part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-Z])", normalized) if part.strip()]

    @staticmethod
    def _word_count(sentence: str) -> int:
        """Count words in a sentence."""
        return len(re.findall(r"\b\w+\b", sentence))

    @staticmethod
    def _count_syllables(word: str) -> int:
        """Estimate syllable count for an English word."""
        cleaned = re.sub(r"[^a-z]", "", (word or "").lower())
        if not cleaned:
            return 0

        vowel_groups = re.findall(r"[aeiouy]+", cleaned)
        syllables = len(vowel_groups)

        if cleaned.endswith("e") and not cleaned.endswith(("le", "ye")) and syllables > 1:
            syllables -= 1

        return max(1, syllables)

    def _compute_flesch_reading_ease(self, draft: str) -> float:
        """Compute Flesch Reading Ease (0-100+ where higher is easier)."""
        sentences = self._split_sentences(draft)
        words = re.findall(r"\b[a-zA-Z]+\b", draft or "")

        if not sentences or not words:
            return 0.0

        syllable_total = sum(self._count_syllables(word) for word in words)
        words_per_sentence = len(words) / len(sentences)
        syllables_per_word = syllable_total / len(words)

        fre = 206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word)
        return round(fre, 2)

    @staticmethod
    def _is_claim_like_sentence(sentence: str) -> bool:
        """Heuristic detector for factual/claim-like statements."""
        normalized = (sentence or "").strip().lower()
        if not normalized:
            return False

        metric_pattern = r"\b\d+(?:\.\d+)?\s*(?:%|gb/s|ghz|mb|tb|w|kw|tflops|cores?|gb|mb/s|hz)?\b"
        comparative_pattern = (
            r"\b(?:up to|increase|decrease|faster|slower|more than|less than|supports|"
            r"support|includes|include|delivers|enable|enables|improves|improved)\b"
        )

        has_metric = re.search(metric_pattern, normalized, flags=re.IGNORECASE) is not None
        has_citation = re.search(r"\[(\d+)\]", normalized) is not None
        has_comparative = re.search(comparative_pattern, normalized, flags=re.IGNORECASE) is not None

        return has_metric or has_citation or has_comparative

    def _compute_content_metrics(self, draft: str, context: dict) -> dict[str, Any]:
        """Compute readability, research-claim coverage, and claim-density."""
        sentences = self._split_sentences(draft)
        claim_like_sentences = [sentence for sentence in sentences if self._is_claim_like_sentence(sentence)]

        claim_density_ratio = (
            len(claim_like_sentences) / len(sentences)
            if sentences
            else 0.0
        )

        source_claims = []
        sources = context.get("sources", []) if isinstance(context, dict) else []
        for source in sources:
            if not isinstance(source, dict):
                continue
            extracted_claim = str(source.get("extracted_claim", "")).strip()
            if extracted_claim and extracted_claim.lower() != "no factual claim":
                source_claims.append(extracted_claim)

        coverage_percentage = 0.0
        covered_claims = 0
        if source_claims and claim_like_sentences:
            source_claim_emb = self.embedding_model.encode(source_claims, convert_to_numpy=True)
            draft_claim_emb = self.embedding_model.encode(claim_like_sentences, convert_to_numpy=True)
            similarity_matrix = cosine_similarity(source_claim_emb, draft_claim_emb)

            for claim_similarities in similarity_matrix:
                if float(np.max(claim_similarities)) >= 0.70:
                    covered_claims += 1

            coverage_percentage = (covered_claims / len(source_claims)) * 100.0

        readability_fre = self._compute_flesch_reading_ease(draft)
        readability_clamped = min(max(readability_fre, 0.0), 100.0)

        metrics = {
            "readability_fre": round(readability_fre, 2),
            "readability_score": round(readability_clamped / 100.0, 4),
            "coverage_percentage": round(coverage_percentage, 2),
            "coverage_score": round(min(max(coverage_percentage, 0.0), 100.0) / 100.0, 4),
            "claim_density_percentage": round(claim_density_ratio * 100.0, 2),
            "claim_density_score": round(claim_density_ratio, 4),
            "claim_like_sentences": len(claim_like_sentences),
            "total_sentences": len(sentences),
            "research_claims_total": len(source_claims),
            "research_claims_covered": covered_claims,
        }

        self._log(
            "content_metrics",
            "Computed readability, coverage, and claim density",
            **metrics,
        )
        return metrics

    @staticmethod
    def _extract_specs(text: str) -> dict[str, set[str]]:
        """Extract numeric specs while avoiding citation/id noise."""
        normalized = text or ""
        normalized = re.sub(r"\[\d+\]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized)

        unit_metrics = {
            f"{value} {unit.lower()}".strip()
            for value, unit in re.findall(
                r"\b(\d+(?:\.\d+)?)\s*(gb/s|ghz|mhz|mb|gb|tb|w|kw|tflops|cores?|%|mah|hz|nits?|mm|cm|ms|fps)\b",
                normalized,
                flags=re.IGNORECASE,
            )
        }

        dimension_metrics = {
            f"{left}x{right}"
            for left, right in re.findall(r"\b(\d{3,4})\s*[x×]\s*(\d{3,4})\b", normalized)
        }

        currency_metrics = {
            f"{currency}{value}"
            for currency, value in re.findall(r"([£$€])\s?(\d{2,5}(?:\.\d+)?)", normalized)
        }

        plain_large_numbers = {
            value
            for value in re.findall(r"\b(\d{3,5})\b", normalized)
        }

        metrics = unit_metrics | dimension_metrics | currency_metrics | plain_large_numbers
        return {"general_metrics": metrics}

    @staticmethod
    def _build_source_text(source: dict) -> str:
        """Build canonical searchable text from one source record."""
        return " ".join(
            [
                str(source.get("subtopic", "")),
                str(source.get("extracted_claim", "")),
                str(source.get("raw_chunk", "")),
            ]
        ).strip()

    def _source_texts(self, context: dict) -> list[str]:
        """Collect source texts from context payload."""
        sources = context.get("sources", []) if isinstance(context, dict) else []
        return [self._build_source_text(src) for src in sources if isinstance(src, dict)]

    @staticmethod
    def _extract_citation_instances(draft: str) -> list[tuple[int, str]]:
        """Return list of (citation_id, containing_sentence) for each [N]."""
        instances: list[tuple[int, str]] = []
        for sentence in EditorAgent._split_sentences(draft):
            for token in re.findall(r"\[(\d+)\]", sentence):
                instances.append((int(token), sentence))
        return instances

    def _select_relevant_sources(self, claim_text: str, context: dict, limit: int = 3) -> list[str]:
        """Pick top source snippets by lexical overlap for LLM prompts."""
        words = set(re.findall(r"\b[a-z0-9]{3,}\b", claim_text.lower()))
        if not words:
            return self._source_texts(context)[:limit]

        scored: list[tuple[int, str]] = []
        for text in self._source_texts(context):
            src_words = set(re.findall(r"\b[a-z0-9]{3,}\b", text.lower()))
            overlap = len(words & src_words)
            scored.append((overlap, text))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [text for _, text in scored[:limit]]


@lru_cache(maxsize=1)
def _get_editor_agent() -> EditorAgent:
    """Create and cache a singleton EditorAgent instance."""
    return EditorAgent(groq_api_key=os.getenv("GROQ_API_KEY"))


def evaluation_agent_node(state: PipelineState) -> dict:
    """Pipeline node wrapper that runs the EditorAgent and adapts output schema."""
    draft_payload = state.get("draft_article") or {}
    research_payload = state.get("research_data") or {}
    revision_count = state.get("revision_count", 0)

    draft_text = str(draft_payload.get("content_md") or "")
    draft_title = str(draft_payload.get("title") or "Untitled")

    try:
        editor = _get_editor_agent()
        result = editor.evaluate(draft=draft_text, context_data=research_payload)
        content_metrics = editor._compute_content_metrics(draft=draft_text, context=research_payload)

        decision = result["decision"]
        status = "APPROVED" if decision == "PUBLISH" else "NEEDS_REVISION"

        accuracy_score = round(float(result["accuracy"]["score"]) / 100.0, 4)
        citation_quality_score = round(float(result["citation"]["citation_score"]) / 100.0, 4)
        readability_score = content_metrics["readability_score"]
        coverage_score = content_metrics["coverage_score"]
        claim_density_score = content_metrics["claim_density_score"]

        remarks = [
            f"Decision: {decision}",
            f"Reason: {result['hard_reason']}",
            f"Factual Accuracy: {result['accuracy']['score']:.2f}/100",
            f"Citation Quality: {result['citation']['citation_score']:.2f}/100",
            f"Readability (Flesch): {content_metrics['readability_fre']:.2f}",
            f"Coverage: {content_metrics['coverage_percentage']:.2f}%",
            f"Claim Density: {content_metrics['claim_density_percentage']:.2f}%",
        ]

        rewrite_suggestions = [
            *result.get("accuracy_feedback", []),
            *result.get("citation_feedback", []),
        ]

        concise_suggestions: list[str] = []
        accuracy_layer1 = result.get("accuracy", {}).get("layer1", {})
        accuracy_layer2 = result.get("accuracy", {}).get("layer2", {})
        citation_data = result.get("citation", {})

        if accuracy_layer1.get("hallucinated_specs"):
            concise_suggestions.append(
                "Remove or re-cite unsupported numeric metrics not present in research sources."
            )
        if accuracy_layer2.get("hallucinations"):
            concise_suggestions.append(
                "Rewrite low-support claims to closely match source evidence and keep inline citations."
            )
        if accuracy_layer2.get("ambiguous_claims"):
            concise_suggestions.append(
                "Clarify ambiguous statements with explicit, source-backed wording."
            )
        if citation_data.get("mis_citations"):
            concise_suggestions.append(
                "Fix citation mapping so each sentence points to the most relevant source ID."
            )
        if content_metrics.get("readability_fre", 100.0) < 45.0:
            concise_suggestions.append(
                "Improve readability by shortening long sentences and removing redundant phrases."
            )

        if concise_suggestions:
            rewrite_suggestions = concise_suggestions[:5]

        if not rewrite_suggestions:
            rewrite_suggestions = ["No blocking issues found"]

        evaluation: Evaluation = {
            "scores": {
                "accuracy": accuracy_score,
                "citation_quality": citation_quality_score,
                "readability": readability_score,
                "coverage": coverage_score,
                "claim_density": claim_density_score,
                "factual": accuracy_score,
                "seo": citation_quality_score,
            },
            "status": status,
            "remarks": remarks,
            "rewrite_suggestions": rewrite_suggestions,
        }

        print(f"[EVAL AGENT] Status: {status}")
        print(
            f"[EVAL AGENT] Scores - Accuracy: {evaluation['scores']['accuracy']}, "
            f"Citation Quality: {evaluation['scores']['citation_quality']}, "
            f"Readability: {evaluation['scores']['readability']}, "
            f"Coverage: {evaluation['scores']['coverage']}, "
            f"Claim Density: {evaluation['scores']['claim_density']}"
        )

        debug_payload = {
            "metadata": {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "agent": "Agent3 Editor",
                "draft_title": draft_title,
                "revision_count": revision_count,
            },
            "inputs": {
                "draft_chars": len(draft_text),
                "source_count": len(research_payload.get("sources", [])) if isinstance(research_payload, dict) else 0,
            },
            "decision": {
                "status": status,
                "decision": decision,
                "reason": result.get("hard_reason", ""),
            },
            "scores": {
                "accuracy": result["accuracy"]["score"],
                "citation_quality": result["citation"]["citation_score"],
                "readability": content_metrics["readability_fre"],
                "coverage": content_metrics["coverage_percentage"],
                "claim_density": content_metrics["claim_density_percentage"],
            },
            "content_metrics": content_metrics,
            "accuracy": result.get("accuracy", {}),
            "citation": result.get("citation", {}),
            "feedback": {
                "accuracy_feedback": result.get("accuracy_feedback", []),
                "citation_feedback": result.get("citation_feedback", []),
            },
            "trace": result.get("trace", []),
        }

        with open("agent3_debug_output.json", "w", encoding="utf-8") as debug_file:
            json.dump(debug_payload, debug_file, ensure_ascii=False, indent=2)

    except Exception as error:
        print(f"[EVAL AGENT] ERROR: {error}")
        evaluation = {
            "scores": {
                "accuracy": 0.0,
                "citation_quality": 0.0,
                "factual": 0.0,
                "seo": 0.0,
                "readability": 0.0,
                "coverage": 0.0,
                "claim_density": 0.0,
            },
            "status": "NEEDS_REVISION",
            "remarks": [f"Evaluation failed: {error}"],
            "rewrite_suggestions": ["Retry evaluation after fixing runtime/dependency issues"],
        }

        error_payload = {
            "metadata": {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "agent": "Agent3 Editor",
                "draft_title": draft_title,
                "revision_count": revision_count,
            },
            "error": str(error),
            "status": "NEEDS_REVISION",
            "scores": {
                "accuracy": 0.0,
                "citation_quality": 0.0,
                "readability": 0.0,
                "coverage": 0.0,
                "claim_density": 0.0,
            },
        }
        with open("agent3_debug_output.json", "w", encoding="utf-8") as debug_file:
            json.dump(error_payload, debug_file, ensure_ascii=False, indent=2)

    print("[EVAL AGENT] ✓ Checkpoint will be saved after this node")
    return {"evaluation": evaluation}
