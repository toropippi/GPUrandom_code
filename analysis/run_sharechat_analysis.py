#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset
from scipy.stats import fisher_exact

DATASET = "tucnguyen/ShareChat"
PLATFORMS = ("claude", "chatgpt")
LANGUAGES = ("Japanese", "English")
ASSISTANT_ROLES = {"assistant", "llm"}
USER_ROLES = {"user", "human"}
SEED = 20260827
RNG = random.Random(SEED)
OUT = Path("output")
OUT.mkdir(parents=True, exist_ok=True)

NONSPACE_RE = re.compile(r"\s+")
JA_RE = re.compile(
    r"[\u3040-\u309f\u30a0-\u30ff\u31f0-\u31ff"
    r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3005\u3006\u303b]"
)
EN_WORD_RE = re.compile(r"\b[A-Za-z]+(?:['’-][A-Za-z]+)*\b")
CODE_RE = re.compile(r"```.*?```", re.S)

MARKERS: dict[str, list[tuple[str, re.Pattern[str]]]] = {
    "Japanese": [
        ("tadashi", re.compile("ただし")),
        ("shikashi", re.compile("しかし")),
        ("tohaie", re.compile("とはいえ")),
        ("mottomo", re.compile("もっとも")),
        ("ippoude", re.compile("一方で")),
        ("chuui", re.compile("注意(?:が必要|してください|しましょう|すべき|点|事項|を要する)?")),
        ("ryuui", re.compile("留意(?:してください|しましょう|が必要|すべき|点|事項)?")),
        ("reigai", re.compile("例外(?:として|的に|があります|がある|を除く|を除き)?")),
    ],
    "English": [
        ("however", re.compile(r"\bhowever\b", re.I)),
        ("that_said", re.compile(r"\bthat said\b", re.I)),
        ("having_said_that", re.compile(r"\bhaving said that\b", re.I)),
        ("nevertheless", re.compile(r"\bnevertheless\b", re.I)),
        ("nonetheless", re.compile(r"\bnonetheless\b", re.I)),
        ("note_that", re.compile(r"\b(?:please\s+)?note that\b", re.I)),
        ("keep_in_mind", re.compile(r"\bkeep in mind(?: that)?\b", re.I)),
        ("bear_in_mind", re.compile(r"\bbear in mind(?: that)?\b", re.I)),
        ("important_to_note", re.compile(r"\b(?:it is|it's) important to note(?: that)?\b", re.I)),
        ("worth_noting", re.compile(r"\b(?:it is|it's) worth noting(?: that)?\b", re.I)),
        ("caveat", re.compile(r"\bcaveats?\b", re.I)),
        ("subject_to", re.compile(r"\bsubject to\b", re.I)),
        ("provided_that", re.compile(r"\bprovided that\b", re.I)),
        ("except_that", re.compile(r"\bexcept that\b", re.I)),
    ],
}
PRIMARY = {"Japanese": "tadashi", "English": "however"}
BROAD = {
    "Japanese": {m for m, _ in MARKERS["Japanese"]},
    "English": {m for m, _ in MARKERS["English"]},
}


def exposure(text: str, language: str) -> int:
    return len(JA_RE.findall(text)) if language == "Japanese" else len(EN_WORD_RE.findall(text))


def code_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(len(m.group(0)) for m in CODE_RE.finditer(text)) / len(text)


def sentence_initial(text: str, start: int) -> int:
    if start <= 0:
        return 1
    prefix = re.sub(r"[\s>*_`#\-–—•]+$", "", text[:start])
    return int(not prefix or bool(re.search(r"[。！？.!?：:;；]$", prefix)))


def snippet(text: str, start: int, end: int, radius: int = 700) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    value = re.sub(r"\s+", " ", text[left:right]).strip()
    return ("…" if left else "") + value + ("…" if right < len(text) else "")


def marker_hits(text: str, language: str) -> tuple[Counter[str], list[dict[str, Any]]]:
    counts: Counter[str] = Counter()
    hits: list[dict[str, Any]] = []
    for marker_id, pattern in MARKERS[language]:
        for match in pattern.finditer(text):
            counts[marker_id] += 1
            hits.append(
                {
                    "marker_id": marker_id,
                    "start": match.start(),
                    "end": match.end(),
                    "matched": match.group(0),
                }
            )
    return counts, hits


def wilson(k: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return math.nan, math.nan
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return max(0.0, center - half), min(1.0, center + half)


def rate_ratio(ca: int, ea: int, cb: int, eb: int) -> tuple[float, float, float]:
    correction = 0.5 if ca == 0 or cb == 0 else 0.0
    aa, bb = ca + correction, cb + correction
    rr = (aa / ea) / (bb / eb)
    se = math.sqrt(1 / aa + 1 / bb)
    z = 1.959963984540054
    return rr, math.exp(math.log(rr) - z * se), math.exp(math.log(rr) + z * se)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    started = time.time()
    diagnostics: dict[str, Counter[str]] = defaultdict(Counter)
    aggregate: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    marker_summary: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    conversations: dict[tuple[str, str, str], dict[str, Any]] = {}
    last_user: dict[tuple[str, str], str] = {}
    dedupe: set[tuple[str, str, str, str]] = set()
    nohit_reservoir: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    nohit_seen: Counter[tuple[str, str]] = Counter()

    response_fields = [
        "platform", "language", "conversation_id", "message_index", "turns_count", "topic", "model",
        "nonspace_chars", "japanese_script_chars", "english_words", "exposure", "code_ratio",
        "primary_count", "broad_count", "has_primary", "has_broad", "primary_initial_count",
    ]
    hit_fields = [
        "platform", "language", "conversation_id", "message_index", "turns_count", "topic", "model",
        "marker_id", "is_primary", "matched", "sentence_initial", "response_nonspace_chars",
        "response_exposure", "response_code_ratio", "preceding_user", "context",
    ]
    nohit_fields = [
        "platform", "language", "conversation_id", "message_index", "turns_count", "topic", "model",
        "response_nonspace_chars", "response_exposure", "response_code_ratio", "preceding_user", "response_text",
    ]

    with gzip.open(OUT / "response_metrics.csv.gz", "wt", encoding="utf-8-sig", newline="") as responses_file, \
         gzip.open(OUT / "hit_contexts.csv.gz", "wt", encoding="utf-8-sig", newline="") as hits_file:
        response_writer = csv.DictWriter(responses_file, fieldnames=response_fields)
        hit_writer = csv.DictWriter(hits_file, fieldnames=hit_fields)
        response_writer.writeheader()
        hit_writer.writeheader()

        for platform in PLATFORMS:
            print(f"Loading {DATASET}/{platform}", flush=True)
            dataset = load_dataset(DATASET, platform, split="train", streaming=True)
            for row_index, row in enumerate(dataset, start=1):
                diagnostics[platform]["rows_total"] += 1
                role = str(row.get("role") or "").strip().lower()
                conversation_id = str(row.get("url") or f"{platform}:row:{row_index}")
                message_index = str(row.get("message_index") if row.get("message_index") is not None else "")
                text = str(row.get("plain_text") or "")
                if not text.strip():
                    diagnostics[platform]["empty_text"] += 1
                    continue

                duplicate_key = (platform, conversation_id, message_index, role + "\0" + text)
                if duplicate_key in dedupe:
                    diagnostics[platform]["duplicate"] += 1
                    continue
                dedupe.add(duplicate_key)

                if role in USER_ROLES:
                    last_user[(platform, conversation_id)] = re.sub(r"\s+", " ", text).strip()[-2500:]
                    diagnostics[platform]["user_rows"] += 1
                    continue
                if role not in ASSISTANT_ROLES:
                    diagnostics[platform]["other_role"] += 1
                    continue

                language = str(row.get("detected_language_final") or "")
                if language not in LANGUAGES:
                    diagnostics[platform]["other_language"] += 1
                    continue

                exp = exposure(text, language)
                if exp <= 0:
                    diagnostics[platform]["zero_exposure"] += 1
                    continue
                diagnostics[platform][f"assistant_{language}"] += 1

                nonspace = len(NONSPACE_RE.sub("", text))
                ja_chars = len(JA_RE.findall(text))
                en_words = len(EN_WORD_RE.findall(text))
                c_ratio = code_ratio(text)
                counts, hits = marker_hits(text, language)
                primary_id = PRIMARY[language]
                primary_count = counts[primary_id]
                broad_count = sum(counts[marker_id] for marker_id in BROAD[language])
                primary_initial = sum(
                    sentence_initial(text, hit["start"])
                    for hit in hits
                    if hit["marker_id"] == primary_id
                )
                turns = row.get("turns_count")
                topic = str(row.get("topic") or "")
                model = str(row.get("model") or row.get("version") or "")

                response_writer.writerow(
                    {
                        "platform": platform,
                        "language": language,
                        "conversation_id": conversation_id,
                        "message_index": message_index,
                        "turns_count": turns,
                        "topic": topic,
                        "model": model,
                        "nonspace_chars": nonspace,
                        "japanese_script_chars": ja_chars,
                        "english_words": en_words,
                        "exposure": exp,
                        "code_ratio": f"{c_ratio:.8f}",
                        "primary_count": primary_count,
                        "broad_count": broad_count,
                        "has_primary": int(primary_count > 0),
                        "has_broad": int(broad_count > 0),
                        "primary_initial_count": primary_initial,
                    }
                )

                agg = aggregate[(platform, language)]
                agg["responses"] += 1
                agg["exposure"] += exp
                agg["nonspace_chars"] += nonspace
                agg["primary_occurrences"] += primary_count
                agg["broad_occurrences"] += broad_count
                agg["responses_primary"] += int(primary_count > 0)
                agg["responses_broad"] += int(broad_count > 0)
                agg["primary_initial"] += primary_initial
                agg["code_heavy"] += int(c_ratio >= 0.5)

                for marker_id, _ in MARKERS[language]:
                    marker_count = counts[marker_id]
                    summary = marker_summary[(platform, language, marker_id)]
                    summary["responses"] += 1
                    summary["exposure"] += exp
                    summary["occurrences"] += marker_count
                    summary["responses_with_marker"] += int(marker_count > 0)

                conv_key = (platform, language, conversation_id)
                conv = conversations.setdefault(
                    conv_key,
                    {
                        "platform": platform,
                        "language": language,
                        "conversation_id": conversation_id,
                        "turns_count": turns,
                        "topic": topic,
                        "model": model,
                        "assistant_responses": 0,
                        "exposure": 0,
                        "nonspace_chars": 0,
                        "primary_count": 0,
                        "broad_count": 0,
                        "code_heavy_responses": 0,
                    },
                )
                conv["assistant_responses"] += 1
                conv["exposure"] += exp
                conv["nonspace_chars"] += nonspace
                conv["primary_count"] += primary_count
                conv["broad_count"] += broad_count
                conv["code_heavy_responses"] += int(c_ratio >= 0.5)
                if not conv["topic"] and topic:
                    conv["topic"] = topic
                if not conv["model"] and model:
                    conv["model"] = model

                preceding_user = last_user.get((platform, conversation_id), "")
                for hit in hits:
                    hit_writer.writerow(
                        {
                            "platform": platform,
                            "language": language,
                            "conversation_id": conversation_id,
                            "message_index": message_index,
                            "turns_count": turns,
                            "topic": topic,
                            "model": model,
                            "marker_id": hit["marker_id"],
                            "is_primary": int(hit["marker_id"] == primary_id),
                            "matched": hit["matched"],
                            "sentence_initial": sentence_initial(text, hit["start"]),
                            "response_nonspace_chars": nonspace,
                            "response_exposure": exp,
                            "response_code_ratio": f"{c_ratio:.8f}",
                            "preceding_user": preceding_user,
                            "context": snippet(text, hit["start"], hit["end"]),
                        }
                    )

                if broad_count == 0:
                    reservoir_key = (platform, language)
                    nohit_seen[reservoir_key] += 1
                    item = {
                        "platform": platform,
                        "language": language,
                        "conversation_id": conversation_id,
                        "message_index": message_index,
                        "turns_count": turns,
                        "topic": topic,
                        "model": model,
                        "response_nonspace_chars": nonspace,
                        "response_exposure": exp,
                        "response_code_ratio": f"{c_ratio:.8f}",
                        "preceding_user": preceding_user,
                        "response_text": re.sub(r"\s+", " ", text).strip()[:6000],
                    }
                    reservoir = nohit_reservoir[reservoir_key]
                    if len(reservoir) < 500:
                        reservoir.append(item)
                    else:
                        position = RNG.randrange(nohit_seen[reservoir_key])
                        if position < 500:
                            reservoir[position] = item

                if row_index % 50000 == 0:
                    print(f"{platform}: {row_index:,} rows", flush=True)

    conversation_rows: list[dict[str, Any]] = []
    for conv in conversations.values():
        conversation_rows.append(
            {
                **conv,
                "has_primary": int(conv["primary_count"] > 0),
                "has_broad": int(conv["broad_count"] > 0),
            }
        )
    with gzip.open(OUT / "conversation_metrics.csv.gz", "wt", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(conversation_rows[0]))
        writer.writeheader()
        writer.writerows(conversation_rows)

    nohit_rows = [item for key in sorted(nohit_reservoir) for item in nohit_reservoir[key]]
    with gzip.open(OUT / "nohit_sample.csv.gz", "wt", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=nohit_fields)
        writer.writeheader()
        writer.writerows(nohit_rows)

    aggregate_rows: list[dict[str, Any]] = []
    for platform in PLATFORMS:
        for language in LANGUAGES:
            agg = aggregate[(platform, language)]
            convs = [
                conv
                for conv in conversation_rows
                if conv["platform"] == platform and conv["language"] == language
            ]
            response_ci = wilson(agg["responses_primary"], agg["responses"])
            conv_primary = sum(conv["has_primary"] for conv in convs)
            conv_ci = wilson(conv_primary, len(convs))
            aggregate_rows.append(
                {
                    "platform": platform,
                    "language": language,
                    "responses": agg["responses"],
                    "conversations": len(convs),
                    "primary_marker": PRIMARY[language],
                    "primary_occurrences": agg["primary_occurrences"],
                    "primary_per_100k_units": agg["primary_occurrences"] * 100000 / agg["exposure"],
                    "primary_response_incidence": agg["responses_primary"] / agg["responses"],
                    "primary_response_ci_low": response_ci[0],
                    "primary_response_ci_high": response_ci[1],
                    "primary_conversation_incidence": conv_primary / len(convs),
                    "primary_conversation_ci_low": conv_ci[0],
                    "primary_conversation_ci_high": conv_ci[1],
                    "primary_sentence_initial_share": (
                        agg["primary_initial"] / agg["primary_occurrences"]
                        if agg["primary_occurrences"]
                        else ""
                    ),
                    "broad_occurrences": agg["broad_occurrences"],
                    "broad_per_100k_units": agg["broad_occurrences"] * 100000 / agg["exposure"],
                    "broad_response_incidence": agg["responses_broad"] / agg["responses"],
                    "broad_conversation_incidence": sum(conv["has_broad"] for conv in convs) / len(convs),
                    "exposure": agg["exposure"],
                    "exposure_unit": "Japanese_script_chars" if language == "Japanese" else "English_words",
                    "nonspace_chars": agg["nonspace_chars"],
                    "code_heavy_responses": agg["code_heavy"],
                }
            )
    write_csv(OUT / "aggregate_summary.csv", aggregate_rows)

    marker_rows: list[dict[str, Any]] = []
    for (platform, language, marker_id), summary in sorted(marker_summary.items()):
        marker_rows.append(
            {
                "platform": platform,
                "language": language,
                "marker_id": marker_id,
                "occurrences": summary["occurrences"],
                "responses_with_marker": summary["responses_with_marker"],
                "responses": summary["responses"],
                "exposure": summary["exposure"],
                "occurrences_per_100k_units": summary["occurrences"] * 100000 / summary["exposure"],
                "response_incidence": summary["responses_with_marker"] / summary["responses"],
            }
        )
    write_csv(OUT / "marker_summary.csv", marker_rows)

    comparison_rows: list[dict[str, Any]] = []
    for language in LANGUAGES:
        for marker_set in ("primary", "broad"):
            chat = aggregate[("chatgpt", language)]
            claude = aggregate[("claude", language)]
            ca = chat[f"{marker_set}_occurrences"]
            cb = claude[f"{marker_set}_occurrences"]
            rr, rr_low, rr_high = rate_ratio(ca, chat["exposure"], cb, claude["exposure"])
            chat_present = chat[f"responses_{marker_set}"]
            claude_present = claude[f"responses_{marker_set}"]
            response_fisher = fisher_exact(
                [
                    [chat_present, chat["responses"] - chat_present],
                    [claude_present, claude["responses"] - claude_present],
                ],
                alternative="two-sided",
            )
            chat_convs = [
                conv for conv in conversation_rows
                if conv["platform"] == "chatgpt" and conv["language"] == language
            ]
            claude_convs = [
                conv for conv in conversation_rows
                if conv["platform"] == "claude" and conv["language"] == language
            ]
            chat_conv_present = sum(conv[f"has_{marker_set}"] for conv in chat_convs)
            claude_conv_present = sum(conv[f"has_{marker_set}"] for conv in claude_convs)
            conversation_fisher = fisher_exact(
                [
                    [chat_conv_present, len(chat_convs) - chat_conv_present],
                    [claude_conv_present, len(claude_convs) - claude_conv_present],
                ],
                alternative="two-sided",
            )
            comparison_rows.append(
                {
                    "language": language,
                    "marker_set": marker_set,
                    "chatgpt_occurrences": ca,
                    "claude_occurrences": cb,
                    "chatgpt_rate_per_100k": ca * 100000 / chat["exposure"],
                    "claude_rate_per_100k": cb * 100000 / claude["exposure"],
                    "rate_ratio_chatgpt_over_claude": rr,
                    "rate_ratio_ci_low": rr_low,
                    "rate_ratio_ci_high": rr_high,
                    "chatgpt_response_incidence": chat_present / chat["responses"],
                    "claude_response_incidence": claude_present / claude["responses"],
                    "response_fisher_odds_ratio": response_fisher.statistic,
                    "response_fisher_p": response_fisher.pvalue,
                    "chatgpt_conversation_incidence": chat_conv_present / len(chat_convs),
                    "claude_conversation_incidence": claude_conv_present / len(claude_convs),
                    "conversation_fisher_odds_ratio": conversation_fisher.statistic,
                    "conversation_fisher_p": conversation_fisher.pvalue,
                }
            )
    write_csv(OUT / "unadjusted_comparisons.csv", comparison_rows)

    metadata = {
        "dataset": DATASET,
        "seed": SEED,
        "elapsed_seconds": time.time() - started,
        "diagnostics": {platform: dict(values) for platform, values in diagnostics.items()},
        "conversation_records": len(conversation_rows),
        "definitions": {
            "Japanese_primary": "exact substring ただし in assistant plain_text",
            "English_primary": "case-insensitive word-boundary however in assistant plain_text",
            "thinking_fields_included": False,
            "Japanese_exposure": "Japanese script characters",
            "English_exposure": "English words",
        },
    }
    (OUT / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
