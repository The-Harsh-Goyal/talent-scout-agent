import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer, util


# ── Lazy-load embedding model (avoids blocking Streamlit on import) ────────────
_EMBEDDING_MODEL = None


def get_embedding_model() -> SentenceTransformer:
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        print("[Matcher] Loading sentence transformer model...")
        _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        print("[Matcher] Model loaded.")
    return _EMBEDDING_MODEL


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — RULE-BASED SCORING
# ══════════════════════════════════════════════════════════════════════════════


def compute_skill_overlap(
    candidate_skills: str,
    must_have: List[str],
    good_to_have: List[str]
) -> Dict:
    candidate_skill_list = [s.strip().lower() for s in str(candidate_skills).split(",")]
    must_lower = [s.strip().lower() for s in must_have]
    good_lower = [s.strip().lower() for s in good_to_have]

    must_matched = [s for s in must_lower if s in candidate_skill_list]
    good_matched = [s for s in good_lower if s in candidate_skill_list]

    must_score = len(must_matched) / len(must_lower) if must_lower else 0
    good_score = len(good_matched) / len(good_lower) if good_lower else 0

    return {
        "must_matched": must_matched,
        "good_matched": good_matched,
        "must_score": round(must_score, 4),
        "good_score": round(good_score, 4),
        "missing_must": [s for s in must_lower if s not in candidate_skill_list],
        "missing_good": [s for s in good_lower if s not in candidate_skill_list]
    }


def compute_experience_score(
    candidate_exp: float,
    min_exp: float,
    max_exp: Optional[float] = None
) -> float:
    try:
        candidate_exp = float(candidate_exp)
        min_exp = float(min_exp)
    except (ValueError, TypeError):
        return 0.5

    # Below minimum — penalise proportionally
    if candidate_exp < min_exp:
        shortfall = min_exp - candidate_exp
        penalty = min(shortfall / max(min_exp, 1), 1.0)
        return round(1.0 - penalty, 4)

    # Overqualified check
    if max_exp:
        try:
            max_exp = float(max_exp)
            if candidate_exp > max_exp + 3:
                return 0.7
        except (ValueError, TypeError):
            pass

    # Meets or exceeds minimum — score between 0.85 and 1.0
    # based on how far above minimum the candidate is
    bonus = min((candidate_exp - min_exp) / max(min_exp, 5), 0.15)
    return round(min(1.0, 0.85 + bonus), 4)


def compute_location_score(
    candidate_location: str,
    jd_location: str,
    candidate_work_mode: str,
    jd_work_mode: str
) -> float:
    c_loc = str(candidate_location).strip().lower()
    j_loc = str(jd_location).strip().lower()
    c_mode = str(candidate_work_mode).strip().lower()
    j_mode = str(jd_work_mode).strip().lower()

    if "remote" in c_mode or "remote" in j_mode:
        return 1.0
    if c_loc == j_loc:
        return 1.0
    elif c_loc in j_loc or j_loc in c_loc:
        return 0.8
    else:
        return 0.4


def compute_domain_score(
    candidate_domain: str,
    jd_domains: List[str]
) -> float:
    if not jd_domains:
        return 0.5

    candidate_domains = [d.strip().lower() for d in str(candidate_domain).split(",")]
    jd_domains_lower = [d.strip().lower() for d in jd_domains]

    matched = [
        d for d in jd_domains_lower
        if any(d in cd or cd in d for cd in candidate_domains)
    ]
    return round(len(matched) / len(jd_domains_lower), 4)


def compute_rule_based_score(candidate: Dict, parsed_jd: Dict) -> Dict:
    skill_overlap = compute_skill_overlap(
        candidate.get("skills", ""),
        parsed_jd.get("must_have_skills", []),
        parsed_jd.get("good_to_have_skills", [])
    )

    exp_score = compute_experience_score(
        candidate.get("years_experience", 0),
        parsed_jd.get("min_years_experience", 0),
        parsed_jd.get("max_years_experience", None)
    )

    loc_score = compute_location_score(
        candidate.get("location", ""),
        parsed_jd.get("location", ""),
        candidate.get("work_mode", ""),
        parsed_jd.get("work_mode", "")
    )

    domain_score = compute_domain_score(
        candidate.get("domain_experience", ""),
        parsed_jd.get("domain_experience", [])
    )

    # Weighted rule-based score (weights sum to 1.0)
    raw_score = (
        skill_overlap["must_score"] * 0.45 +
        skill_overlap["good_score"] * 0.15 +
        exp_score                   * 0.20 +
        loc_score                   * 0.10 +
        domain_score                * 0.10
    )

    return {
        "rule_score": round(min(raw_score, 1.0), 4),
        "skill_overlap": skill_overlap,
        "exp_score": exp_score,
        "loc_score": loc_score,
        "domain_score": domain_score
    }


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — SEMANTIC SCORING
# ══════════════════════════════════════════════════════════════════════════════


def build_candidate_profile_text(candidate: Dict) -> str:
    """
    Combine candidate fields into one searchable text block
    for semantic embedding comparison.
    """
    parts = [
        str(candidate.get("current_title", "")),
        str(candidate.get("summary", "")),
        str(candidate.get("skills", "")),
        str(candidate.get("domain_experience", "")),
        str(candidate.get("current_company", "")),
        f"{candidate.get('years_experience', '')} years experience",
        str(candidate.get("location", "")),
        str(candidate.get("work_mode", ""))
    ]
    return " | ".join([p for p in parts if p.strip()])


def build_jd_text(parsed_jd: Dict) -> str:
    """
    Combine parsed JD fields into one searchable text block
    for semantic embedding comparison.
    """
    parts = [
        str(parsed_jd.get("job_title", "")),
        str(parsed_jd.get("summary", "")),
        " ".join(parsed_jd.get("must_have_skills", [])),
        " ".join(parsed_jd.get("good_to_have_skills", [])),
        " ".join(parsed_jd.get("domain_experience", [])),
        " ".join(parsed_jd.get("keywords", [])),
        str(parsed_jd.get("seniority", "")),
        str(parsed_jd.get("location", "")),
        str(parsed_jd.get("work_mode", ""))
    ]
    return " | ".join([p for p in parts if p.strip()])


def compute_semantic_scores(
    candidates: List[Dict],
    parsed_jd: Dict
) -> List[float]:
    """
    Batch compute cosine similarity between JD embedding
    and all candidate profile embeddings.
    Returns a list of floats in [0, 1] in the same order as candidates.
    """
    model = get_embedding_model()

    jd_text = build_jd_text(parsed_jd)
    candidate_texts = [build_candidate_profile_text(c) for c in candidates]

    jd_embedding = model.encode(jd_text, convert_to_tensor=True)
    candidate_embeddings = model.encode(candidate_texts, convert_to_tensor=True)

    cosine_scores = util.cos_sim(jd_embedding, candidate_embeddings)[0]
    return [round(float(score), 4) for score in cosine_scores]


# ══════════════════════════════════════════════════════════════════════════════
# EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════


def build_explainability(
    candidate: Dict,
    skill_overlap: Dict,
    exp_score: float,
    loc_score: float,
    domain_score: float,
    semantic_score: float,
    final_score: float
) -> str:
    parts = []

    if skill_overlap["must_matched"]:
        parts.append(f"Matched must-have skills: {', '.join(skill_overlap['must_matched'])}.")
    if skill_overlap["missing_must"]:
        parts.append(f"Missing must-have skills: {', '.join(skill_overlap['missing_must'])}.")
    if skill_overlap["good_matched"]:
        parts.append(f"Bonus preferred skills: {', '.join(skill_overlap['good_matched'])}.")

    if exp_score >= 1.0:
        parts.append(f"Experience is a strong fit ({candidate.get('years_experience')} yrs).")
    elif exp_score >= 0.7:
        parts.append(f"Experience is acceptable ({candidate.get('years_experience')} yrs).")
    else:
        parts.append(f"Experience is below required ({candidate.get('years_experience')} yrs).")

    if loc_score == 1.0:
        parts.append("Location/work mode fully aligned.")
    elif loc_score >= 0.7:
        parts.append("Location is a partial match.")
    else:
        parts.append("Location mismatch — may require relocation or mode change.")

    if domain_score >= 0.7:
        parts.append("Strong domain relevance.")
    elif domain_score >= 0.4:
        parts.append("Moderate domain relevance.")
    else:
        parts.append("Limited domain overlap.")

    if semantic_score >= 0.75:
        parts.append(f"High semantic profile-JD alignment ({round(semantic_score * 100, 1)}%).")
    elif semantic_score >= 0.5:
        parts.append(f"Moderate semantic alignment ({round(semantic_score * 100, 1)}%).")
    else:
        parts.append(f"Low semantic alignment ({round(semantic_score * 100, 1)}%).")

    return " ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# HYBRID RANKING — COMBINES BOTH LAYERS
# ══════════════════════════════════════════════════════════════════════════════


def rank_candidates_hybrid(
    df: pd.DataFrame,
    parsed_jd: Dict,
    top_n: int = 10,
    rule_weight: float = 0.65,
    semantic_weight: float = 0.35
) -> pd.DataFrame:
    """
    Ranks candidates using a weighted hybrid of:
    - Rule-based score (skill overlap, experience, location, domain)
    - Semantic similarity score (Sentence Transformer cosine similarity)

    rule_weight + semantic_weight must sum to 1.0
    """
    candidates = df.to_dict(orient="records")

    # Batch compute semantic scores for all candidates at once (efficient)
    print("[Matcher] Computing semantic embeddings...")
    semantic_scores = compute_semantic_scores(candidates, parsed_jd)
    print("[Matcher] Semantic scoring complete.")

    results = []

    for i, candidate in enumerate(candidates):
        rule_result = compute_rule_based_score(candidate, parsed_jd)
        semantic_score = semantic_scores[i]

        rule_score = rule_result["rule_score"]
        skill_overlap = rule_result["skill_overlap"]

        # Hybrid final match score scaled to 100
        final_match_score = round(
            (rule_score * rule_weight + semantic_score * semantic_weight) * 100,
            2
        )

        explainability = build_explainability(
            candidate,
            skill_overlap,
            rule_result["exp_score"],
            rule_result["loc_score"],
            rule_result["domain_score"],
            semantic_score,
            final_match_score
        )

        results.append({
            "candidate_id":       candidate.get("candidate_id"),
            "name":               candidate.get("name"),
            "current_title":      candidate.get("current_title"),
            "years_experience":   candidate.get("years_experience"),
            "location":           candidate.get("location"),
            "work_mode":          candidate.get("work_mode"),
            "current_company":    candidate.get("current_company"),
            "candidate_status":   candidate.get("candidate_status"),
            "notice_period_days": candidate.get("notice_period_days"),
            "expected_salary_lpa":candidate.get("expected_salary_lpa"),
            "final_match_score":  final_match_score,
            "rule_based_score":   round(rule_score * 100, 2),
            "semantic_score":     round(semantic_score * 100, 2),
            "must_skill_score":   round(skill_overlap["must_score"] * 100, 2),
            "good_skill_score":   round(skill_overlap["good_score"] * 100, 2),
            "experience_score":   round(rule_result["exp_score"] * 100, 2),
            "location_score":     round(rule_result["loc_score"] * 100, 2),
            "domain_score":       round(rule_result["domain_score"] * 100, 2),
            "must_matched":       ", ".join(skill_overlap["must_matched"]),
            "missing_must":       ", ".join(skill_overlap["missing_must"]),
            "good_matched":       ", ".join(skill_overlap["good_matched"]),
            "explainability":     explainability
        })

    ranked_df = pd.DataFrame(results)
    ranked_df = ranked_df.sort_values(by="final_match_score", ascending=False)
    ranked_df = ranked_df.reset_index(drop=True)
    ranked_df.index += 1
    ranked_df.index.name = "Rank"

    return ranked_df.head(top_n)