import streamlit as st
import pandas as pd
from src.jd_parser import parse_jd_with_llm, parsed_jd_to_dict
from src.matcher import rank_candidates_hybrid
from src.interest_simulator import (
    simulate_candidate_interest,
    compute_interest_score,
    interest_signals_to_dict,
    CandidateInterestSignals
)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Talent Scout Agent",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Talent Scout Agent")
st.markdown("*AI-powered candidate discovery, matching, and interest simulation for recruiters.*")
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CANDIDATE POOL
# ══════════════════════════════════════════════════════════════════════════════

st.subheader("1. Candidate Pool")

uploaded_file = st.file_uploader(
    "Upload your own candidate CSV (optional)",
    type=["csv"],
    help="CSV must include columns: candidate_id, name, current_title, years_experience, "
         "location, work_mode, skills, domain_experience, current_company, summary, "
         "notice_period_days, expected_salary_lpa, candidate_status."
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded {len(df)} candidates from uploaded file.")
else:
    df = pd.read_csv("data/candidates.csv")
    st.info(f"Using built-in sample pool: {len(df)} candidates.")

with st.expander("Preview Candidate Pool", expanded=False):
    st.dataframe(df, use_container_width=True)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — JD INPUT
# ══════════════════════════════════════════════════════════════════════════════

st.subheader("2. Paste Job Description")

jd_text = st.text_area(
    "Enter the full job description below",
    height=250,
    placeholder="Paste your JD here..."
)

col_left, col_right = st.columns([1, 2])

with col_left:
    top_n = st.slider(
        "Number of candidates to shortlist",
        min_value=3,
        max_value=min(20, len(df)),
        value=5
    )

with col_right:
    st.markdown("")
    st.markdown("")
    st.caption(
        "The agent will parse the JD, score all candidates on match, "
        "simulate outreach for the top shortlist, and produce a final ranked output."
    )

st.divider()

run_button = st.button("▶ Run Talent Scout Agent", type="primary", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — AGENT RUN
# ══════════════════════════════════════════════════════════════════════════════

if run_button:
    if not jd_text.strip():
        st.warning("Please enter a job description before running.")
        st.stop()

    # ── Step 1: Parse JD ──────────────────────────────────────────────────────
    st.subheader("3. Parsed Job Description")

    with st.spinner("Step 1/3 — Parsing job description with LLM..."):
        try:
            parsed = parse_jd_with_llm(jd_text)
        except RuntimeError as e:
            st.error(
                "All AI models are currently rate-limited. "
                "Please wait 30 seconds and try again.\n\n"
                f"Detail: {e}"
            )
            st.stop()

    parsed_dict = parsed_jd_to_dict(parsed)

    with st.expander("Parsed JD — Structured Output", expanded=True):
        st.json(parsed_dict)

    st.divider()

    # ── Step 2: Match Candidates ──────────────────────────────────────────────
    st.subheader("4. Candidate Match Results")
    st.caption("Ranked by hybrid Match Score (rule-based + semantic similarity).")

    with st.spinner("Step 2/3 — Matching and ranking candidates..."):
        ranked_df = rank_candidates_hybrid(df, parsed_dict, top_n=top_n)

    match_display_cols = [
        "name", "current_title", "years_experience", "location",
        "final_match_score", "rule_based_score", "semantic_score",
        "must_skill_score", "experience_score",
        "must_matched", "missing_must", "explainability"
    ]
    st.dataframe(ranked_df[match_display_cols], use_container_width=True)

    st.divider()

    # ── Step 3: Interest Simulation ───────────────────────────────────────────
    st.subheader("5. Interest Simulation")
    st.caption("Each shortlisted candidate receives a simulated outreach and interest score.")

    interest_results = []
    progress    = st.progress(0)
    status_text = st.empty()
    top_candidates = ranked_df.head(top_n).copy()

    for i, (_, row) in enumerate(top_candidates.iterrows()):

        # Safe candidate lookup with guard
        match = df[df["candidate_id"] == row["candidate_id"]]
        if match.empty:
            print(f"[App] candidate_id {row['candidate_id']} not found in source df. Skipping.")
            continue
        candidate_dict = match.iloc[0].to_dict()

        status_text.text(
            f"Simulating interest for {row['name']} ({i + 1}/{len(top_candidates)})..."
        )

        try:
            signals      = simulate_candidate_interest(candidate_dict, parsed_dict)
            interest_score = compute_interest_score(signals, candidate_dict, parsed_dict)
        except RuntimeError as e:
            st.warning(f"Rate limit hit for {row['name']} — using fallback scores.")
            signals        = CandidateInterestSignals()
            interest_score = 50.0

        interest_results.append({
            "candidate_id":           row["candidate_id"],
            "name":                   row["name"],
            "current_title":          row["current_title"],
            "years_experience":       row["years_experience"],
            "location":               row["location"],
            "current_company":        row["current_company"],
            "candidate_status":       row["candidate_status"],
            "notice_period_days":     row["notice_period_days"],
            "expected_salary_lpa":    row["expected_salary_lpa"],
            "match_score":            row["final_match_score"],
            "rule_based_score":       row["rule_based_score"],
            "semantic_score":         row["semantic_score"],
            "must_matched":           row["must_matched"],
            "missing_must":           row["missing_must"],
            "interest_score":         interest_score,
            "interest_level":         signals.interest_level,
            "sentiment":              signals.sentiment,
            "compensation_alignment": signals.compensation_alignment,
            "work_mode_alignment":    signals.work_mode_alignment,
            "follow_up_likelihood":   signals.follow_up_likelihood,
            "simulated_response":     signals.simulated_response,
            "key_signals":            ", ".join(signals.key_signals),
            "interest_summary":       signals.summary,
            "explainability":         row["explainability"]
        })

        progress.progress((i + 1) / len(top_candidates))

    status_text.empty()
    progress.empty()

    st.divider()

    # ── Final Shortlist ───────────────────────────────────────────────────────
    if not interest_results:
        st.error("No candidates were processed. Please check your candidate CSV and try again.")
        st.stop()

    final_df = pd.DataFrame(interest_results)

    final_df["combined_score"] = (
        0.6 * final_df["match_score"] + 0.4 * final_df["interest_score"]
    ).round(2)

    final_df = final_df.sort_values(
        by="combined_score", ascending=False
    ).reset_index(drop=True)
    final_df.index     += 1
    final_df.index.name = "Rank"

    def recommend_action(row):
        if row["combined_score"] >= 75:
            return "Call Now"
        elif row["combined_score"] >= 55:
            return "Schedule Screen"
        elif row["combined_score"] >= 35:
            return "Nurture"
        else:
            return "Deprioritise"

    final_df["recruiter_action"] = final_df.apply(recommend_action, axis=1)

    # ── Shortlist Table ───────────────────────────────────────────────────────
    st.subheader("6. Final Ranked Shortlist")

    shortlist_display = [
        "name", "current_title", "years_experience", "location",
        "match_score", "rule_based_score", "semantic_score",
        "interest_score", "combined_score",
        "interest_level", "follow_up_likelihood",
        "compensation_alignment", "work_mode_alignment",
        "recruiter_action", "must_matched", "missing_must",
        "key_signals", "simulated_response", "interest_summary",
        "explainability"
    ]

    st.dataframe(final_df[shortlist_display], use_container_width=True)

    st.divider()

    # ── Per-candidate Detail Expanders ────────────────────────────────────────
    st.subheader("7. Candidate Detail View")

    action_emoji = {
        "Call Now":        "🟢",
        "Schedule Screen": "🔵",
        "Nurture":         "🟡",
        "Deprioritise":    "🔴"
    }

    for _, row in final_df.iterrows():
        emoji = action_emoji.get(row["recruiter_action"], "⚪")

        with st.expander(
            f"{emoji} {row['name']} — Combined Score: {row['combined_score']} | {row['recruiter_action']}",
            expanded=False
        ):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Match Score",    f"{row['match_score']}")
                st.metric("Interest Score", f"{row['interest_score']}")
                st.metric("Combined Score", f"{row['combined_score']}")

            with col2:
                st.markdown(f"**Interest Level:** {row['interest_level']}")
                st.markdown(f"**Sentiment:** {row['sentiment']}")
                st.markdown(f"**Follow-up Likelihood:** {row['follow_up_likelihood']}")
                st.markdown(f"**Compensation Alignment:** {row['compensation_alignment']}")
                st.markdown(f"**Work Mode Alignment:** {row['work_mode_alignment']}")

            with col3:
                st.markdown(f"**Notice Period:** {row['notice_period_days']} days")
                st.markdown(f"**Expected Salary:** {row['expected_salary_lpa']} LPA")
                st.markdown(f"**Status:** {row['candidate_status']}")
                st.markdown(f"**Company:** {row['current_company']}")

            st.markdown("**Match Explainability:**")
            st.info(row["explainability"])

            st.markdown("**Key Interest Signals:**")
            st.warning(row["key_signals"])

            st.markdown("**Simulated Candidate Response:**")
            st.success(row["simulated_response"])

            st.markdown("**Interest Summary:**")
            st.write(row["interest_summary"])

    st.divider()

    # ── Export ────────────────────────────────────────────────────────────────
    st.subheader("8. Export Shortlist")

    csv_export = final_df[shortlist_display].to_csv(index=True).encode("utf-8")

    st.download_button(
        label="⬇ Download Shortlist as CSV",
        data=csv_export,
        file_name="talent_scout_shortlist.csv",
        mime="text/csv",
        use_container_width=True
    )