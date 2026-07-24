from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search , scrape_url 
from dotenv import load_dotenv

load_dotenv()

#model setup 
llm = ChatOpenAI(model = "gpt-4o",temperature=0)
llm_critic = ChatOpenAI(model = "gpt-4o",temperature=0)

#1st agent 
def build_search_agent():
    return create_agent(
        model = llm,
        tools= [web_search]
    )

def build_reader_agent():
    return create_agent(
        model = llm,
        tools = [scrape_url]
    )


RESEARCH_RUBRIC = """
1. Research quality and source credibility
0: Sources are missing, unsuitable, or not connected to the claims.
1: Some credible evidence is used, but coverage, diversity, or citation
   traceability is limited.
2: Important claims are traceable to diverse, authoritative, relevant sources.

2. Accuracy and factual grounding
0: Major factual errors, fabricated information, or unsupported claims.
1: Generally plausible, but some claims are vague, overstated, or insufficiently supported.
2: Important claims are supported, accurately represented, and appropriately qualified.

3. Depth of analysis
0: Mostly description or summary.
1: Some interpretation is included, but analysis remains generic or uneven.
2: Meaningful comparisons, causes, implications, trade-offs, limitations, or
   conflicting perspectives are analyzed.

4. Completeness and relevance
0: Major parts of the topic are missing or substantial content is irrelevant.
1: Main aspects are covered, but important dimensions remain incomplete.
2: The report thoroughly addresses the topic's important dimensions without padding.

5. Structure and clarity
0: Disorganized, repetitive, or difficult to understand.
1: Generally clear, but contains unnecessary repetition, weak transitions, or
   unclear sections.
2: Concise, coherent, logically organized, and easy to follow.
"""

#writer chain
writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an evidence-grounded research writer.

Write analytical, accurate, and clearly structured reports using only the
research material provided.

Rules:
1. Do not invent facts, statistics, dates, quotations, or sources.
2. Every important factual or quantitative claim must be supported by a
   source contained in the research.
3. Use source markers such as [S1], [S2], and [S3] after supported claims.
4. Distinguish established facts from interpretations, estimates, and predictions.
5. Explain comparisons, implications, tradeoffs, limitations, and areas of uncertainty.
6. Do not add content merely to make the report longer.
7. When the research is insufficient, explicitly state the limitation rather
   than filling the gap with unsupported information.
"""
    ),
    (
        "human",
        """
Write a research report answering the following topic.

Topic:
{topic}

Research evidence:
{research}

Required structure:

1. Introduction
   - Define the subject and scope.
   - Explain why the topic matters.

2. Key Findings
   - Organize findings into meaningful subsections.
   - Explain the evidence, significance, implications, and limitations.
   - Include comparisons or contrasting perspectives where supported.

3. Research Limitations
   - Identify missing, uncertain, conflicting, or limited evidence.

4. Conclusion
   - Answer the research question directly.
   - Summarize the most important evidence-supported insights.

5. Sources
   - List each source marker, title, publisher, and URL.

Return only the completed report.
"""
    ),
])

writer_chain = writer_prompt | llm | StrOutputParser()


#critic_chain
critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        f"""
You are a rigorous, evidence-grounded research evaluator.

Evaluate the report only against:
1. The research topic.
2. The supplied research evidence.
3. The scoring rubric below.

Do not assume that a claim is true merely because it sounds plausible.
Do not reward length, formatting, confident language, or the number of URLs.

For every category:
- Assign a score from 0 to 2.
- Give a specific reason.
- Reference an example from the report.
- Explain the exact change needed to earn the next score.
- State whether the improvement requires additional research.

Rubric:
{RESEARCH_RUBRIC}

The total score must equal the sum of the five category scores.

Important:
- A writing revision cannot solve missing research.
- When evidence is missing, mark the improvement as requiring additional research.
- Do not increase a score simply because the report follows the requested headings.
"""
    ),
    (
        "human",
        """
Topic:
{topic}

Research evidence available to the writer:
{research}

Report being evaluated:
{report}

Respond in this exact format:

Score: X/10

Score Breakdown:
- Research quality and source credibility: X/2
  Reason:
  Next improvement:
  Requires additional research: Yes/No

- Accuracy and factual grounding: X/2
  Reason:
  Next improvement:
  Requires additional research: Yes/No

- Depth of analysis: X/2
  Reason:
  Next improvement:
  Requires additional research: Yes/No

- Completeness and relevance: X/2
  Reason:
  Next improvement:
  Requires additional research: Yes/No

- Structure and clarity: X/2
  Reason:
  Next improvement:
  Requires additional research: Yes/No

Strengths:
- ...
- ...

Prioritized Improvements:
1. ...
2. ...
3. ...

One-line verdict:
...
"""
    ),
])

critic_chain = critic_prompt | llm_critic | StrOutputParser()

# revision writer chain
revision_writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an evidence-grounded research editor.

Revise the report using the critic feedback and supplied research evidence.

Rules:
1. Preserve accurate and well-supported strengths from the previous report.
2. Address the highest-priority weaknesses first.
3. Use only facts contained in the research evidence.
4. Do not invent statistics, sources, dates, quotations, or examples.
5. Keep or add source markers such as [S1] after factual claims.
6. Do not blindly expand the report. Remove repetition and unsupported material.
7. When feedback requests evidence that is unavailable, state the research
   limitation instead of fabricating an answer.
8. Ensure that improvements in one category do not reduce clarity or introduce
   unsupported claims.
"""
    ),
    (
        "human",
        """
Topic:
{topic}

Research evidence:
{research}

Previous report:
{previous_report}

Critic feedback:
{feedback}

Produce a revised report that addresses the prioritized improvements.

Required structure:
1. Introduction
2. Key Findings
3. Research Limitations
4. Conclusion
5. Sources

Return only the revised report.
"""
    ),
])

revision_writer_chain = revision_writer_prompt | llm | StrOutputParser()

revision_critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        f"""
You are a rigorous research evaluator reviewing a revised report.

Your primary task is to determine whether the revised report meaningfully
improved upon the previous report and addressed the previous feedback.

Do not introduce completely new evaluation standards unless the revised report
contains a serious new factual or structural problem.

For each previously identified improvement:
1. State whether it was fully addressed, partially addressed, or not addressed.
2. Cite a specific change in the revised report.
3. Explain why the category score should increase, remain unchanged, or decrease.

Use the same five-category rubric:
{RESEARCH_RUBRIC}

The total score must equal the sum of the five category scores.

A score should increase only when the revised report materially satisfies the
previously stated next improvement.

A score should remain unchanged when the revision is superficial or incomplete.

Do not keep a score unchanged merely because the report is not perfect.
A score of 2 means the category is strong and meets the rubric, not that no
further improvement is possible.
"""
    ),
    (
        "human",
        """
Topic:
{topic}

Research evidence:
{research}

Previous report:
{previous_report}

Previous critic feedback:
{previous_feedback}

Revised report:
{revised_report}

Respond in this exact format:

Previous Feedback Resolution:
1. Improvement:
   Status: Fully addressed / Partially addressed / Not addressed
   Evidence from revised report:
2. Improvement:
   Status:
   Evidence from revised report:
3. Improvement:
   Status:
   Evidence from revised report:

Score: X/10

Score Breakdown:
- Research quality and source credibility: X/2
  Change from previous evaluation:
  Reason:

- Accuracy and factual grounding: X/2
  Change from previous evaluation:
  Reason:

- Depth of analysis: X/2
  Change from previous evaluation:
  Reason:

- Completeness and relevance: X/2
  Change from previous evaluation:
  Reason:

- Structure and clarity: X/2
  Change from previous evaluation:
  Reason:

Remaining highest-priority improvement:
...
"""
    ),
])

revision_critic_chain = (
    revision_critic_prompt
    | llm_critic
    | StrOutputParser()
)