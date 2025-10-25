import streamlit as st
import asyncio
from agent.loop import run_insight_scout

st.set_page_config(page_title="Insight Scout", layout="wide")
st.title("🔎 Insight Scout - Autonomous Research Agent")

q = st.text_input("Topic", value="electric bikes this week")
run = st.button("Run Agent")

if run:
    with st.spinner("Thinking… scraping… clustering… summarizing…"):
        result = asyncio.run(run_insight_scout(q))

    st.subheader("Research Brief")
    st.markdown(result.get("report", "(no report)"))

    st.subheader("Themes")
    themes = result.get("themes", [])
    if not themes:
        st.info("No themes found.")
    for t in themes:
        with st.expander(f"Score {t['score']} - Theme"):
            st.markdown(t.get("summary", ""))
            st.markdown("**Sources:**\n" + "\n".join([f"- {u}" for u in t.get("sources", [])]))
