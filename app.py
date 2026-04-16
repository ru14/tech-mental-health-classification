"""
Mental Health in Tech — Streamlit MVP
======================================
Multi-page app:  Home → Login/Register → Survey → Results

Run:
    streamlit run app.py
"""

import os
import sqlite3
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from passlib.context import CryptContext
from sklearn.pipeline import Pipeline

# ── Constants ─────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).parent
MODEL_PATH = _ROOT / "models" / "rf_pipeline.pkl"
DB_PATH = _ROOT / "users.db"

FEATURE_COLS = [
    "Age", "Gender", "self_employed", "family_history",
    "work_interfere", "no_employees", "remote_work", "tech_company",
    "benefits", "care_options", "wellness_program", "seek_help",
    "anonymity", "mental_health_consequence", "coworkers",
    "supervisor", "mental_vs_physical", "obs_consequence",
]
CATEGORICAL_COLS = [c for c in FEATURE_COLS if c != "Age"]

SURVEY_QUESTIONS = {
    "Age": {
        "label": "🎂 How old are you?",
        "type": "number",
        "min": 16, "max": 99, "default": 28,
    },
    "Gender": {
        "label": "🧑 What is your gender?",
        "type": "select",
        "options": ["Male", "Female", "Other"],
    },
    "self_employed": {
        "label": "💼 Are you self-employed?",
        "type": "select",
        "options": ["No", "Yes"],
    },
    "family_history": {
        "label": "🧬 Do you have a family history of mental illness?",
        "type": "select",
        "options": ["No", "Yes"],
    },
    "work_interfere": {
        "label": "📉 How often does a mental health condition interfere with your work?",
        "type": "select",
        "options": ["Never", "Rarely", "Sometimes", "Often"],
    },
    "no_employees": {
        "label": "🏢 How many employees does your company have?",
        "type": "select",
        "options": ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"],
    },
    "remote_work": {
        "label": "🏠 Do you work remotely at least 50% of the time?",
        "type": "select",
        "options": ["No", "Yes"],
    },
    "tech_company": {
        "label": "💻 Is your employer primarily a tech company?",
        "type": "select",
        "options": ["Yes", "No"],
    },
    "benefits": {
        "label": "🏥 Does your employer provide mental health benefits?",
        "type": "select",
        "options": ["Yes", "No", "Don't know"],
    },
    "care_options": {
        "label": "ℹ️ Do you know the mental health care options your employer provides?",
        "type": "select",
        "options": ["Yes", "No", "Not sure"],
    },
    "wellness_program": {
        "label": "🧘 Has your employer discussed mental health as part of a wellness program?",
        "type": "select",
        "options": ["Yes", "No", "Don't know"],
    },
    "seek_help": {
        "label": "🔍 Does your employer provide resources to seek mental health help?",
        "type": "select",
        "options": ["Yes", "No", "Don't know"],
    },
    "anonymity": {
        "label": "🔒 Is your anonymity protected when seeking mental health support?",
        "type": "select",
        "options": ["Yes", "No", "Don't know"],
    },
    "mental_health_consequence": {
        "label": "⚠️ Could discussing mental health with your employer have negative consequences?",
        "type": "select",
        "options": ["No", "Maybe", "Yes"],
    },
    "coworkers": {
        "label": "🤝 Would you discuss a mental health issue with your coworkers?",
        "type": "select",
        "options": ["Yes", "Some of them", "No"],
    },
    "supervisor": {
        "label": "👔 Would you discuss a mental health issue with your direct supervisor?",
        "type": "select",
        "options": ["Yes", "Some of them", "No"],
    },
    "mental_vs_physical": {
        "label": "⚖️ Does your employer take mental health as seriously as physical health?",
        "type": "select",
        "options": ["Yes", "No", "Don't know"],
    },
    "obs_consequence": {
        "label": "👁️ Have you observed negative consequences for colleagues with mental health conditions?",
        "type": "select",
        "options": ["No", "Yes"],
    },
}

FEATURE_LABELS = {k: v["label"] for k, v in SURVEY_QUESTIONS.items()}

# ── Database helpers ──────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    conn = _get_conn()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    UNIQUE NOT NULL,
            email         TEXT    UNIQUE NOT NULL,
            password_hash TEXT    NOT NULL,
            created_at    TEXT    DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS survey_responses (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       INTEGER NOT NULL,
            answers       TEXT    NOT NULL,
            prediction    REAL    NOT NULL,
            submitted_at  TEXT    DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        """
    )
    conn.commit()
    conn.close()


_PWD_CONTEXT = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _hash_password(password: str) -> str:
    return _PWD_CONTEXT.hash(password)


def _verify_password(password: str, password_hash: str) -> bool:
    return _PWD_CONTEXT.verify(password, password_hash)


def register_user(username: str, email: str, password: str) -> tuple[bool, str]:
    pw_hash = _hash_password(password)
    try:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username.strip(), email.strip().lower(), pw_hash),
        )
        conn.commit()
        conn.close()
        return True, "Account created! You can now log in."
    except sqlite3.IntegrityError as e:
        if "username" in str(e):
            return False, "Username already taken."
        if "email" in str(e):
            return False, "Email already registered."
        return False, "Registration failed."


def login_user(username: str, password: str) -> tuple[bool, dict | None]:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM users WHERE username = ?", (username.strip(),)
    ).fetchone()
    conn.close()
    if row is None:
        return False, None
    if _verify_password(password, row["password_hash"]):
        return True, dict(row)
    return False, None


def save_response(user_id: int, answers: dict, prediction: float) -> None:
    conn = _get_conn()
    conn.execute(
        "INSERT INTO survey_responses (user_id, answers, prediction) VALUES (?, ?, ?)",
        (user_id, json.dumps(answers), float(prediction)),
    )
    conn.commit()
    conn.close()


def get_history(user_id: int) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM survey_responses WHERE user_id = ? ORDER BY submitted_at DESC",
        (user_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Model helpers ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading prediction model…")
def load_model() -> Pipeline:
    if not MODEL_PATH.exists():
        st.info("First-time setup: training model on your data (≈ 30 s)…")
        import sys
        sys.path.insert(0, str(_ROOT / "src"))
        from train_model import train
        train()
    return joblib.load(MODEL_PATH)


def predict(pipeline, answers: dict) -> tuple[float, pd.Series]:
    """Return (probability_of_treatment, feature_importances_series)."""
    row = pd.DataFrame([answers])[FEATURE_COLS]
    prob = pipeline.predict_proba(row)[0][1]
    rf = pipeline.named_steps["classifier"]
    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS)
    return prob, importances.sort_values(ascending=False)


# ── UI helpers ────────────────────────────────────────────────────────────────

def _set_page(page: str) -> None:
    st.session_state.page = page
    st.rerun()


def _logout() -> None:
    for k in ["user", "page", "last_result"]:
        st.session_state.pop(k, None)
    st.session_state.page = "home"
    st.rerun()


def _sidebar() -> None:
    with st.sidebar:
        st.image(
            "https://img.shields.io/badge/Mental%20Health%20in%20Tech-Survey%20App-5c67f2?style=for-the-badge",
            use_column_width=True,
        )
        st.markdown("---")
        if st.session_state.get("user"):
            user = st.session_state["user"]
            st.markdown(f"👋 **{user['username']}**")
            if st.button("📋 Take Survey", use_container_width=True):
                _set_page("survey")
            if st.button("📊 My History", use_container_width=True):
                _set_page("history")
            if st.button("🏠 Home", use_container_width=True):
                _set_page("home")
            st.markdown("---")
            if st.button("🚪 Log out", use_container_width=True):
                _logout()
        else:
            if st.button("🔐 Login", use_container_width=True):
                _set_page("login")
            if st.button("✏️ Register", use_container_width=True):
                _set_page("register")
            if st.button("🏠 Home", use_container_width=True):
                _set_page("home")

        st.markdown("---")
        st.caption("Data: [OSMI Survey](https://osmihelp.org/research)")
        st.caption("Built with ❤️ using Streamlit")

        # LinkedIn share CTA
        linkedin_share = (
            "https://www.linkedin.com/sharing/share-offsite/"
            "?url=https%3A%2F%2Fshare.streamlit.io"
        )
        st.link_button("🔗 Share on LinkedIn", linkedin_share, use_container_width=True)


# ── Pages ─────────────────────────────────────────────────────────────────────

def page_home() -> None:
    st.title("🧠 Mental Health in Tech")
    st.subheader("Workplace Mental Health Assessment Tool")

    col1, col2, col3 = st.columns(3)
    col1.metric("Survey Respondents", "1,259")
    col2.metric("Features Analysed", "18")
    col3.metric("Model Accuracy", "~82 %")

    st.markdown(
        """
        ### What is this?
        This tool uses a **Random Forest classifier** trained on the
        [OSMI Mental Health in Tech Survey](https://osmihelp.org/research) to estimate
        how likely a tech-industry employee is to seek mental health treatment, based on
        workplace and demographic factors.

        ### 🔒 Privacy first
        Your responses are stored only under your account and are never shared.
        This tool is **informational only** and is not a substitute for professional
        mental health support.

        ### How it works
        1. **Register / Log in** — create a free account
        2. **Take the survey** — 18 quick questions about your workplace
        3. **See your result** — a personalised likelihood score + top factors driving it

        ---
        """
    )

    st.info(
        "💡 **Disclaimer:** This is a data-driven showcase project. Results are "
        "statistical estimates based on survey patterns — not clinical advice. "
        "If you are struggling, please reach out to a mental health professional."
    )

    col_a, col_b = st.columns(2)
    if col_a.button("🚀 Take the Survey", type="primary", use_container_width=True):
        if st.session_state.get("user"):
            _set_page("survey")
        else:
            _set_page("register")
    if col_b.button("🔐 Login", use_container_width=True):
        _set_page("login")


def page_login() -> None:
    st.title("🔐 Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", type="primary", use_container_width=True)

    if submitted:
        if not username or not password:
            st.warning("Please fill in all fields.")
            return
        ok, user = login_user(username, password)
        if ok:
            st.session_state["user"] = user
            st.success(f"Welcome back, {user['username']}! 👋")
            _set_page("home")
        else:
            st.error("Invalid username or password.")

    st.markdown("---")
    st.markdown("Don't have an account?")
    if st.button("✏️ Register here", use_container_width=True):
        _set_page("register")


def page_register() -> None:
    st.title("✏️ Create Account")
    with st.form("register_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        password2 = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button(
            "Create Account", type="primary", use_container_width=True
        )

    if submitted:
        if not username or not email or not password:
            st.warning("Please fill in all fields.")
            return
        if password != password2:
            st.error("Passwords do not match.")
            return
        if len(password) < 8:
            st.error("Password must be at least 8 characters.")
            return
        ok, msg = register_user(username, email, password)
        if ok:
            st.success(msg)
            _set_page("login")
        else:
            st.error(msg)

    st.markdown("---")
    st.markdown("Already have an account?")
    if st.button("🔐 Login here", use_container_width=True):
        _set_page("login")


def page_survey() -> None:
    if not st.session_state.get("user"):
        st.warning("Please log in to take the survey.")
        _set_page("login")
        return

    st.title("📋 Workplace Mental Health Survey")
    st.caption(
        "Based on the OSMI Mental Health in Tech Survey · 18 questions · ~2 min"
    )
    st.markdown("---")

    pipeline = load_model()
    answers: dict = {}

    with st.form("survey_form"):
        # ── Demographic ──────────────────────────────────────────────────────
        st.subheader("👤 About You")
        c1, c2 = st.columns(2)
        q = SURVEY_QUESTIONS["Age"]
        answers["Age"] = c1.number_input(
            q["label"], min_value=q["min"], max_value=q["max"], value=q["default"]
        )
        q = SURVEY_QUESTIONS["Gender"]
        answers["Gender"] = c2.selectbox(q["label"], q["options"])

        q = SURVEY_QUESTIONS["self_employed"]
        answers["self_employed"] = st.radio(
            q["label"], q["options"], horizontal=True
        )
        q = SURVEY_QUESTIONS["family_history"]
        answers["family_history"] = st.radio(
            q["label"], q["options"], horizontal=True
        )

        # ── Workplace ────────────────────────────────────────────────────────
        st.subheader("🏢 Your Workplace")
        q = SURVEY_QUESTIONS["work_interfere"]
        answers["work_interfere"] = st.select_slider(
            q["label"], options=q["options"]
        )
        c1, c2 = st.columns(2)
        q = SURVEY_QUESTIONS["no_employees"]
        answers["no_employees"] = c1.selectbox(q["label"], q["options"])
        q = SURVEY_QUESTIONS["remote_work"]
        answers["remote_work"] = c2.radio(q["label"], q["options"], horizontal=True)
        q = SURVEY_QUESTIONS["tech_company"]
        answers["tech_company"] = st.radio(q["label"], q["options"], horizontal=True)

        # ── Employer support ─────────────────────────────────────────────────
        st.subheader("🏥 Employer Mental Health Support")
        for key in ["benefits", "care_options", "wellness_program", "seek_help", "anonymity"]:
            q = SURVEY_QUESTIONS[key]
            answers[key] = st.radio(q["label"], q["options"], horizontal=True)

        # ── Workplace culture ────────────────────────────────────────────────
        st.subheader("🗣️ Workplace Culture")
        for key in [
            "mental_health_consequence", "coworkers", "supervisor",
            "mental_vs_physical", "obs_consequence",
        ]:
            q = SURVEY_QUESTIONS[key]
            answers[key] = st.radio(q["label"], q["options"], horizontal=True)

        st.markdown("---")
        submitted = st.form_submit_button(
            "🔮 Get My Result", type="primary", use_container_width=True
        )

    if submitted:
        with st.spinner("Analysing your responses…"):
            prob, importances = predict(pipeline, answers)

        save_response(st.session_state["user"]["id"], answers, prob)
        st.session_state["last_result"] = {"prob": prob, "importances": importances, "answers": answers}
        _set_page("results")


def page_results() -> None:
    if not st.session_state.get("last_result"):
        st.warning("No result to show. Take the survey first!")
        if st.button("📋 Take Survey"):
            _set_page("survey")
        return

    result = st.session_state["last_result"]
    prob: float = result["prob"]
    importances: pd.Series = result["importances"]
    pct = int(round(prob * 100))

    st.title("📊 Your Result")
    st.markdown("---")

    # ── Score gauge ──────────────────────────────────────────────────────────
    if pct >= 65:
        level, color, emoji = "High likelihood", "#e74c3c", "🔴"
        advice = (
            "Your responses suggest a **high likelihood** of benefiting from mental health "
            "support. Consider speaking with a mental health professional or exploring your "
            "employer's EAP (Employee Assistance Programme)."
        )
    elif pct >= 40:
        level, color, emoji = "Moderate likelihood", "#f39c12", "🟡"
        advice = (
            "Your responses suggest a **moderate likelihood**. It may be worth checking in "
            "with your employer's wellness resources or a trusted colleague."
        )
    else:
        level, color, emoji = "Lower likelihood", "#27ae60", "🟢"
        advice = (
            "Your responses suggest a **lower likelihood** at this time. Keep looking after "
            "your wellbeing — small habits matter!"
        )

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(
            f"""
            <div style="text-align:center; padding:24px; border-radius:16px;
                        background:{color}22; border:2px solid {color};">
                <div style="font-size:3.5rem; font-weight:700; color:{color};">{pct}%</div>
                <div style="font-size:1rem; color:{color};">{emoji} {level}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(f"### {emoji} {level}")
        st.info(advice)
        st.caption(
            "⚕️ **Disclaimer:** This score is a statistical estimate based on "
            "survey patterns. It is not a clinical diagnosis."
        )

    st.markdown("---")

    # ── Top factors ──────────────────────────────────────────────────────────
    st.subheader("🔑 Top Factors Driving Your Score")
    top5 = importances.head(5).reset_index()
    top5.columns = ["Feature", "Importance"]
    top5["Question"] = top5["Feature"].map(
        lambda f: FEATURE_LABELS.get(f, f).lstrip("🎂🧑💼🧬📉🏢🏠💻🏥ℹ️🧘🔍🔒⚠️🤝👔⚖️👁️ ")
    )
    top5["Your Answer"] = top5["Feature"].map(
        lambda f: str(result["answers"].get(f, "—"))
    )

    st.bar_chart(top5.set_index("Question")["Importance"], height=260)

    with st.expander("📋 See details"):
        st.dataframe(
            top5[["Question", "Your Answer", "Importance"]]
            .assign(Importance=top5["Importance"].map("{:.3f}".format))
            .reset_index(drop=True),
            use_container_width=True,
        )

    st.markdown("---")

    # ── Recommendations ───────────────────────────────────────────────────────
    st.subheader("💡 Personalised Recommendations")
    answers = result["answers"]
    recs = []

    if answers.get("benefits") in ("No", "Don't know"):
        recs.append(
            "**Ask HR about mental health benefits.** Many employers offer confidential "
            "support that employees aren't aware of."
        )
    if answers.get("seek_help") in ("No", "Don't know"):
        recs.append(
            "**Explore available resources.** Check whether your employer has an EAP, "
            "counselling sessions, or a dedicated wellbeing portal."
        )
    if answers.get("anonymity") in ("No", "Don't know"):
        recs.append(
            "**Know your rights.** In most regions, seeking mental health support is "
            "confidential. Ask HR about your company's privacy policy."
        )
    if answers.get("work_interfere") in ("Sometimes", "Often"):
        recs.append(
            "**Consider speaking to a manager or HR.** If work is consistently affected, "
            "a short conversation can open doors to flexible arrangements."
        )
    if answers.get("mental_health_consequence") in ("Yes", "Maybe"):
        recs.append(
            "**Build psychological safety.** Resources like *Mind* (mind.org.uk) and "
            "*Mental Health America* offer workplace advocacy guides."
        )
    if not recs:
        recs.append(
            "🌟 Your workplace environment looks supportive! Keep using those resources "
            "and check in regularly with your own wellbeing."
        )

    for i, rec in enumerate(recs, 1):
        st.markdown(f"{i}. {rec}")

    st.markdown("---")

    # ── Actions ──────────────────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns(3)
    if col_a.button("🔄 Take Again", use_container_width=True):
        _set_page("survey")
    if col_b.button("📈 My History", use_container_width=True):
        _set_page("history")
    linkedin_share = (
        f"https://www.linkedin.com/sharing/share-offsite/"
        f"?url=https%3A%2F%2Fshare.streamlit.io"
    )
    col_c.link_button("🔗 Share on LinkedIn", linkedin_share, use_container_width=True)


def page_history() -> None:
    if not st.session_state.get("user"):
        _set_page("login")
        return

    st.title("📈 My Survey History")
    user_id = st.session_state["user"]["id"]
    rows = get_history(user_id)

    if not rows:
        st.info("No submissions yet. Take the survey to see your history here!")
        if st.button("📋 Take Survey"):
            _set_page("survey")
        return

    records = []
    for r in rows:
        pct = int(round(r["prediction"] * 100))
        if pct >= 65:
            badge = "🔴 High"
        elif pct >= 40:
            badge = "🟡 Moderate"
        else:
            badge = "🟢 Lower"
        records.append(
            {
                "Date": r["submitted_at"][:16],
                "Score": f"{pct}%",
                "Level": badge,
            }
        )

    df = pd.DataFrame(records)
    st.dataframe(df, use_container_width=True, hide_index=True)

    scores = [int(round(r["prediction"] * 100)) for r in rows]
    if len(scores) > 1:
        st.subheader("Trend over time")
        chart_df = pd.DataFrame(
            {"Score (%)": scores[::-1]},
            index=pd.RangeIndex(1, len(scores) + 1),
        )
        chart_df.index.name = "Submission"
        st.line_chart(chart_df, height=220)

    if st.button("📋 Take New Survey", type="primary"):
        _set_page("survey")


# ── App entry point ───────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Mental Health in Tech",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://www.mind.org.uk/",
            "Report a bug": "https://github.com/ru14/tech-mental-health-classification/issues",
            "About": (
                "**Mental Health in Tech** — a data-driven workplace mental health "
                "assessment tool built with Streamlit and scikit-learn.\n\n"
                "Data: OSMI Mental Health in Tech Survey · Model: Random Forest"
            ),
        },
    )

    _init_db()

    if "page" not in st.session_state:
        st.session_state.page = "home"

    _sidebar()

    page = st.session_state.get("page", "home")
    if page == "home":
        page_home()
    elif page == "login":
        page_login()
    elif page == "register":
        page_register()
    elif page == "survey":
        page_survey()
    elif page == "results":
        page_results()
    elif page == "history":
        page_history()
    else:
        page_home()


if __name__ == "__main__":
    main()
