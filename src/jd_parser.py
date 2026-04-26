import os
import json
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from src.llm_client import chat_with_fallback, OPENROUTER_API_KEY


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

class ParsedJD(BaseModel):
    job_title:            str            = Field(default="")
    seniority:            str            = Field(default="")
    min_years_experience: float          = Field(default=0)
    max_years_experience: Optional[float]= Field(default=None)
    location:             str            = Field(default="")
    work_mode:            str            = Field(default="")
    employment_type:      str            = Field(default="")
    must_have_skills:     List[str]      = Field(default_factory=list)
    good_to_have_skills:  List[str]      = Field(default_factory=list)
    domain_experience:    List[str]      = Field(default_factory=list)
    responsibilities:     List[str]      = Field(default_factory=list)
    keywords:             List[str]      = Field(default_factory=list)
    summary:              str            = Field(default="")


# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
You are an expert recruiting analyst.
Extract a job description into clean structured JSON.

Rules:
- Return ONLY valid JSON. No markdown fences, no explanation.
- Normalize skill names, for example use "SQL" not "Structured Query Language".
- Separate must-have skills from good-to-have skills carefully.
- Estimate min_years_experience only if clearly stated or strongly inferable.
- Use empty strings, null, 0, or empty arrays when not specified.
- Keep summary short and recruiter-friendly (1-2 sentences).
"""


def _build_jd_prompt(jd_text: str) -> str:
    return f"""
Extract the following job description into valid JSON.

Return ONLY JSON. No markdown fences. No explanation.
Use this exact structure:
{{
  "job_title": "",
  "seniority": "",
  "min_years_experience": 0,
  "max_years_experience": null,
  "location": "",
  "work_mode": "",
  "employment_type": "",
  "must_have_skills": [],
  "good_to_have_skills": [],
  "domain_experience": [],
  "responsibilities": [],
  "keywords": [],
  "summary": ""
}}

Job Description:
{jd_text}
"""


# ══════════════════════════════════════════════════════════════════════════════
# CORE PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_jd_with_llm(jd_text: str) -> ParsedJD:

    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is missing. Add it to your .env file.")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": _build_jd_prompt(jd_text)}
    ]

    defaults = {
        "job_title":            "",
        "seniority":            "",
        "min_years_experience": 0,
        "max_years_experience": None,
        "location":             "",
        "work_mode":            "",
        "employment_type":      "",
        "must_have_skills":     [],
        "good_to_have_skills":  [],
        "domain_experience":    [],
        "responsibilities":     [],
        "keywords":             [],
        "summary":              ""
    }

    try:
        content = chat_with_fallback(
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0
        )
    except RuntimeError as e:
        raise RuntimeError(f"All LLM models failed during JD parsing: {e}") from e

    # Final None/empty guard
    if not content or not content.strip():
        print("[JD Parser] LLM returned empty content. Returning empty ParsedJD.")
        return ParsedJD()

    content = content.strip()

    # Strip markdown fences defensively — some models ignore instructions
    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(content)
        defaults.update(data)
    except json.JSONDecodeError:
        print(f"[JD Parser] JSON decode failed. Using defaults. Raw content: {content[:300]}")

    # Ensure list fields are actually lists — guard against LLM returning strings
    list_fields = [
        "must_have_skills",
        "good_to_have_skills",
        "domain_experience",
        "responsibilities",
        "keywords"
    ]
    for field in list_fields:
        if not isinstance(defaults.get(field), list):
            raw = defaults.get(field, "")
            if isinstance(raw, str) and raw.strip():
                # Try to recover comma-separated strings as lists
                defaults[field] = [s.strip() for s in raw.split(",") if s.strip()]
            else:
                defaults[field] = []

    # Ensure numeric fields are actually numeric
    for field in ["min_years_experience"]:
        try:
            defaults[field] = float(defaults[field])
        except (ValueError, TypeError):
            defaults[field] = 0

    if defaults.get("max_years_experience") is not None:
        try:
            defaults["max_years_experience"] = float(defaults["max_years_experience"])
        except (ValueError, TypeError):
            defaults["max_years_experience"] = None

    try:
        return ParsedJD(**defaults)
    except Exception as e:
        print(f"[JD Parser] Pydantic validation failed: {e}. Returning empty ParsedJD.")
        return ParsedJD()


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def parsed_jd_to_dict(jd: ParsedJD) -> dict:
    return jd.model_dump()


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    sample_jd = """
    We are hiring a Senior Data Engineer with 5+ years of experience in Python, SQL, Azure,
    Databricks, and Spark. Experience with Microsoft Fabric is a plus. The role is based in
    Bengaluru and follows a hybrid work model. Candidates with experience building ETL pipelines
    and lakehouse architectures in FinTech or SaaS domains are preferred.
    """
    parsed = parse_jd_with_llm(sample_jd)
    print(parsed.model_dump_json(indent=2))