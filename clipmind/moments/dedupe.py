"""Candidate moment deduplication across temporal overlap and semantic/lexical similarity."""
import re
from typing import List, Dict, Any


def _time_overlap_ratio(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def _text_similarity(a: str, b: str) -> float:
    wa = {w for w in re.findall(r"\w+", (a or "").lower(), flags=re.U) if len(w) >= 3}
    wb = {w for w in re.findall(r"\w+", (b or "").lower(), flags=re.U) if len(w) >= 3}
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def deduplicate_candidates(candidates: List[Dict[str, Any]], max_overlap: float = 0.15,
                           max_text_sim: float = 0.52) -> List[Dict[str, Any]]:
    """Filters out temporally overlapping or near-duplicate transcripts, keeping the highest-scoring clip."""
    sorted_cands = sorted(candidates, key=lambda c: -c["overall_score"])
    selected: List[Dict[str, Any]] = []

    for cand in sorted_cands:
        s, e = cand["start"], cand["end"]
        txt = cand.get("transcript_text", "")
        clash = False
        for ex in selected:
            if _time_overlap_ratio(s, e, ex["start"], ex["end"]) > max_overlap:
                clash = True
                break
            if txt and ex.get("transcript_text") and _text_similarity(txt, ex["transcript_text"]) >= max_text_sim:
                clash = True
                break
        if not clash:
            selected.append(cand)

    return selected
