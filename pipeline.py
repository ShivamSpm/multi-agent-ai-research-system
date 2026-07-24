import re
from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain, revision_writer_chain, revision_critic_chain

MAX_ITERATIONS = 3
SCORE_THRESHOLD = 8

def parse_score(feedback: str) -> float:
    match = re.search(r"Score:\s*(\d+(?:\.\d+)?)\s*/\s*10", feedback, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 0.0  # default ensures loop continues if score can't be parsed

def run_research_pipeline(topic: str) -> dict:

    state = {}

    # step 1 - search agent
    print("\n" + " ="*50)
    print("step 1 - search agent is working ...")
    print("="*50)

    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
    })
    state["search_results"] = search_result['messages'][-1].content
    print("\n search result ", state['search_results'])

    # step 2 - reader agent
    print("\n" + " ="*50)
    print("step 2 - Reader agent is scraping top resources ...")
    print("="*50)

    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
                        "messages": [
                            (
                                "user",
                                f"""
                    You are researching the following topic:

                    {topic}

                    From the search results below:

                    1. Select exactly 3 of the most relevant and credible URLs.
                    2. Call the scrape_url tool separately for each selected URL.
                    3. Do not rely only on the search-result snippets.
                    4. Return the gathered content under clearly labeled source sections.

                    Use this format:

                    SOURCE 1
                    Title:
                    URL:
                    Scraped Content:

                    SOURCE 2
                    Title:
                    URL:
                    Scraped Content:

                    SOURCE 3
                    Title:
                    URL:
                    Scraped Content:

                    Search Results:
                    {state['search_results']}
                    """
                            )
                        ]
                    })
    state['scraped_content'] = reader_result['messages'][-1].content
    print("\nscraped content: \n", state['scraped_content'])

    research_combined = (
        f"SEARCH RESULTS : \n {state['search_results']} \n\n"
        f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
    )

    # steps 3 & 4 - writer/critic revision loop
    print("\n" + " ="*50)
    print("step 3 - Writer/Critic revision loop starting ...")
    print("="*50)

    draft = writer_chain.invoke({
        "topic": topic,
        "research": research_combined
    })
    iteration = 1
    state["iterations"] = [] 

    previous_draft = None
    previous_feedback = None

    while True:
        print(f"\nCritic reviewing draft {iteration} ...")

        # Use the normal critic for the initial draft.
        if iteration == 1:
            feedback = critic_chain.invoke({
                "topic": topic,
                "research": research_combined,
                "report": draft
            })

        # For later drafts, compare the revision against the previous
        # report and the feedback it was supposed to address.
        else:
            feedback = revision_critic_chain.invoke({
                "topic": topic,
                "research": research_combined,
                "previous_report": previous_draft,
                "previous_feedback": previous_feedback,
                "revised_report": draft
            })

        score = parse_score(feedback)

        print(f"Draft {iteration} score: {score}/10")
        print(f"\nCritic feedback for draft {iteration}:\n{feedback}")

        state["iterations"].append({
            "draft": draft,
            "feedback": feedback,
            "score": score
        })

        # Stop when the required score is reached or all evaluations are complete.
        if score >= SCORE_THRESHOLD or iteration >= MAX_ITERATIONS:
            break

        print(
            f"\nScore {score}/10 is below the threshold. "
            "Conducting targeted follow-up research..."
        )

        # Search specifically for evidence requested by the current critic.
        followup_search = search_agent.invoke({
            "messages": [
                (
                    "user",
                    f"""
    Find credible information needed to improve a research report about:

    Topic:
    {topic}

    Critic feedback:
    {feedback}

    Focus specifically on the missing evidence, examples, perspectives,
    comparisons, industry implications, and trade-offs requested by the critic.

    Use the web_search tool once and return exactly 5 relevant sources with:
    - Title
    - URL
    - Snippet

    Do not write the report.
    """
                )
            ]
        })

        followup_search_results = followup_search["messages"][-1].content

        # Scrape the sources that best address the critic's feedback.
        followup_reader = reader_agent.invoke({
            "messages": [
                (
                    "user",
                    f"""
    The following sources were found to improve a research report about:

    Topic:
    {topic}

    Critic feedback:
    {feedback}

    Select exactly 3 sources that best address the critic's requested
    improvements. Call the scrape_url tool separately for each selected URL.

    Return the results under clearly labeled source sections containing:
    - Title
    - URL
    - Scraped Content

    Search results:
    {followup_search_results}
    """
                )
            ]
        })

        followup_scraped_content = followup_reader["messages"][-1].content

        # Add the newly collected evidence to the existing research packet.
        research_combined += (
            f"\n\nFOLLOW-UP RESEARCH FOR DRAFT {iteration + 1}:\n"
            f"{followup_search_results}\n\n"
            f"FOLLOW-UP SCRAPED CONTENT:\n"
            f"{followup_scraped_content}"
        )

        # Save the current report and feedback so the comparative critic can
        # evaluate whether the next revision actually addressed them.
        previous_draft = draft
        previous_feedback = feedback

        print(f"\nWriter revising draft {iteration} ...")

        draft = revision_writer_chain.invoke({
            "topic": topic,
            "research": research_combined,
            "previous_report": previous_draft,
            "feedback": previous_feedback
        })

        iteration += 1

    return state


if __name__ == "__main__":
    topic = input("\n Enter a research topic : ")
    run_research_pipeline(topic)