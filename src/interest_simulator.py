import os
import json
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from src.llm_client import chat_with_fallback, OPENROUTER_API_KEY


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

class CandidateInterestSignals(BaseModel):
    simulated_response:     str            = Field(default="")
    interest_level:         str            = Field(default="medium")
    sentiment:              str            = Field(default="neutral")
    notice_period_days:     Optional[int]  = Field(default=None)
    compensation_alignment: str            = Field(default="unknown")
    work_mode_alignment:    str            = Field(default="unknown")
    follow_up_likelihood:   str            = Field(default="medium")
    key_signals:            List[str]      = Field(default_factory=list)
    summary:                str            = Field(default="")


# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
You are simulating a realistic candidate's response to recruiter outreach.

Return a valid JSON object only. No explanation, no markdown fences.
Use exactly these keys:
simulated_response, interest_level, sentiment, notice_period_days,
compensation_alignment, work_mode_alignment, follow_up_likelihood,
key_signals, summary

Allowed values:
- interest_level: high, medium, low, not_interested
- sentiment: positive, neutral, negative
- compensation_alignment: aligned, negotiable, misaligned, unknown
- work_mode_alignment: aligned, partially_aligned, misaligned, unknown
- follow_up_likelihood: high, medium, low
- notice_period_days: integer or null

Rules:
- Be realistic, not overly enthusiastic unless strongly justified.
- Consider skill fit, seniority, location, work mode, notice period, and compensation.
- key_signals must be a list of short strings explaining the candidate's stance.
- Do not include any extra keys.
"""


def build_interest_prompt(candidate: Dict, parsed_jd: Dict) -> str:
    return f"""
Candidate Profile:
Name: {candidate.get('name', '')}
Current Title: {candidate.get('current_title', '')}
Experience: {candidate.get('years_experience', '')} years
Location: {candidate.get('location', '')}
Work Mode Preference: {candidate.get('work_mode', '')}
Skills: {candidate.get('skills', '')}
Domain Experience: {candidate.get('domain_experience', '')}
Current Company: {candidate.get('current_company', '')}
Summary: {candidate.get('summary', '')}
Notice Period: {candidate.get('notice_period_days', '')} days
Expected Salary: {candidate.get('expected_salary_lpa', '')} LPA
Candidate Status: {candidate.get('candidate_status', '')}

Job Description:
Title: {parsed_jd.get('job_title', '')}
Seniority: {parsed_jd.get('seniority', '')}
Minimum Experience: {parsed_jd.get('min_years_experience', '')} years
Location: {parsed_jd.get('location', '')}
Work Mode: {parsed_jd.get('work_mode', '')}
Must Have Skills: {', '.join(parsed_jd.get('must_have_skills', []))}
Good to Have Skills: {', '.join(parsed_jd.get('good_to_have_skills', []))}
Domain: {', '.join(parsed_jd.get('domain_experience', []))}
Summary: {parsed_jd.get('summary', '')}

Simulate how this candidate would respond to recruiter outreach for this role,
then extract structured engagement signals. Return JSON only.
"""


# ══════════════════════════════════════════════════════════════════════════════
# CORE SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def simulate_candidate_interest(
    candidate: Dict,
    parsed_jd: Dict
) -> CandidateInterestSignals:

    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is missing. Add it to your .env file.")

    prompt = build_interest_prompt(candidate, parsed_jd)

    raw_notice = candidate.get("notice_period_days", None)
    safe_notice = int(raw_notice) if raw_notice is not None else None

    defaults = {
        "simulated_response":     "Thanks for reaching out. I may be open to hearing more depending on role fit and compensation.",
        "interest_level":         "medium",
        "sentiment":              "neutral",
        "notice_period_days":     safe_notice,
        "compensation_alignment": "unknown",
        "work_mode_alignment":    "unknown",
        "follow_up_likelihood":   "medium",
        "key_signals":            ["Fallback used — LLM response unavailable"],
        "summary":                "Candidate shows moderate interest but needs more discussion."
    }

    try:
        content = chat_with_fallback(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        # Strip markdown fences defensively — some models ignore instructions
        content = content.strip()
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()

        data = json.loads(content)
        defaults.update(data)

    except json.JSONDecodeError as e:
        print(f"[Interest Simulator] JSON decode failed for {candidate.get('name', '?')}: {e}")
    except RuntimeError as e:
        print(f"[Interest Simulator] All models failed for {candidate.get('name', '?')}: {e}")
        raise
    except Exception as e:
        print(f"[Interest Simulator] Unexpected error for {candidate.get('name', '?')}: {e}")

    # Safe cast notice period — LLM may return float like 30.0
    if defaults.get("notice_period_days") is not None:
        try:
            defaults["notice_period_days"] = int(defaults["notice_period_days"])
        except (ValueError, TypeError):
            defaults["notice_period_days"] = safe_notice

    # Validate allowed enum values — guard against LLM hallucinating bad values
    allowed = {
        "interest_level":         {"high", "medium", "low", "not_interested"},
        "sentiment":              {"positive", "neutral", "negative"},
        "compensation_alignment": {"aligned", "negotiable", "misaligned", "unknown"},
        "work_mode_alignment":    {"aligned", "partially_aligned", "misaligned", "unknown"},
        "follow_up_likelihood":   {"high", "medium", "low"}
    }

    for field, valid_values in allowed.items():
        if defaults.get(field) not in valid_values:
            print(f"[Interest Simulator] Invalid value for {field}: '{defaults.get(field)}'. Resetting to default.")
            defaults[field] = {
                "interest_level":         "medium",
                "sentiment":              "neutral",
                "compensation_alignment": "unknown",
                "work_mode_alignment":    "unknown",
                "follow_up_likelihood":   "medium"
            }[field]

    # Ensure key_signals is always a list of strings
    if not isinstance(defaults.get("key_signals"), list):
        defaults["key_signals"] = ["Signal parsing failed — fallback applied"]

    try:
        return CandidateInterestSignals(**defaults)
    except Exception as e:
        print(f"[Interest Simulator] Pydantic validation failed: {e}. Using safe fallback.")
        return CandidateInterestSignals(
            notice_period_days=safe_notice,
            key_signals=["Validation fallback triggered"]
        )


# ══════════════════════════════════════════════════════════════════════════════
# INTEREST SCORE FORMULA
# ══════════════════════════════════════════════════════════════════════════════

def compute_interest_score(
    signals: CandidateInterestSignals,
    candidate: Dict,
    parsed_jd: Dict
) -> float:
    """
    Deterministic interest score formula.
    Max achievable = 35 + 15 + 20 + 15 + 10 + 5 = 100
    """
    score = 0.0

    # Interest level — 35 pts max
    interest_map = {"high": 35, "medium": 22, "low": 10, "not_interested": 0}
    score += interest_map.get(signals.interest_level, 0)

    # Sentiment — 15 pts max
    sentiment_map = {"positive": 15, "neutral": 8, "negative": 0}
    score += sentiment_map.get(signals.sentiment, 0)

    # Notice period / availability — 20 pts max
    notice = signals.notice_period_days
    if notice is not None:
        if notice <= 30:
            score += 20
        elif notice <= 60:
            score += 12
        else:
            score += 5
    else:
        score += 8  # Unknown — give partial credit

    # Compensation alignment — 15 pts max
    comp_map = {"aligned": 15, "negotiable": 10, "misaligned": 0, "unknown": 6}
    score += comp_map.get(signals.compensation_alignment, 6)

    # Work mode alignment — 10 pts max
    mode_map = {"aligned": 10, "partially_aligned": 5, "misaligned": 0, "unknown": 4}
    score += mode_map.get(signals.work_mode_alignment, 4)

    # Follow-up likelihood — 5 pts max
    follow_map = {"high": 5, "medium": 3, "low": 0}
    score += follow_map.get(signals.follow_up_likelihood, 0)

    return round(min(score, 100.0), 2)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def interest_signals_to_dict(signals: CandidateInterestSignals) -> Dict:
    return signals.model_dump()