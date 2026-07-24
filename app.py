import streamlit as st

from pipeline import run_research_pipeline


st.set_page_config(page_title="Multi-Agent AI Research System", layout="wide")
st.title("Multi-Agent AI Research System")
st.write("Enter a topic and run the research pipeline.")

with st.form("research_form"):
    topic = st.text_input("Research topic", placeholder="e.g. Future of quantum computing")
    submitted = st.form_submit_button("Run Research")

if submitted:
    clean_topic = topic.strip()

    if not clean_topic:
        st.warning("Please enter a topic before running the pipeline.")
    else:
        with st.spinner("Running agents and generating report..."):
            try:
                results = run_research_pipeline(clean_topic)
            except Exception as exc:
                st.error(f"Pipeline failed: {exc}")
            else:
                st.success("Research pipeline completed.")

                # with this:
                iterations = results.get("iterations", [])

                for i, it in enumerate(iterations[:-1], start=1):
                    with st.expander(f"Draft {i} — Score: {it['score']}/10"):
                        st.markdown("**Report**")
                        st.write(it["draft"])
                        st.markdown("**Critic Feedback**")
                        st.write(it["feedback"])

                if iterations:
                    final = iterations[-1]
                    st.subheader(f"Final Report (Draft {len(iterations)}) — Score: {final['score']}/10")
                    st.write(final["draft"])
                    st.subheader("Final Critic Feedback")
                    st.write(final["feedback"])
                else:
                    st.write("No report generated.")

                with st.expander("Search Results"):
                    st.text(results.get("search_results", "No search results available."))

                with st.expander("Scraped Content"):
                    st.text(results.get("scraped_content", "No scraped content available."))
