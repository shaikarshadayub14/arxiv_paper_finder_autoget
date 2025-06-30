from autogen_agentchat.agents import AssistantAgent
# from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.ollama import OllamaChatCompletionClient
import os
import asyncio
from autogen_agentchat.teams import RoundRobinGroupChat
import arxiv
from typing import List,Dict,AsyncGenerator
from dotenv import load_dotenv

load_dotenv()

# openai_brain = OpenAIChatCompletionClient(model='gpt-4o',api_key=os.getenv('OPENAI_API_KEY'))

autogen_brain = OllamaChatCompletionClient(model='qwen3:0.6b')


def arxiv_search(query: str, max_results: int = 5) -> List[Dict]:
    """Return a compact list of arXiv papers matching *query*.

    Each element contains: ``title``, ``authors``, ``published``, ``summary`` and
    ``pdf_url``.  The helper is wrapped as an AutoGen *FunctionTool* below so it
    can be invoked by agents through the normal tool‑use mechanism.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    papers: List[Dict] = []
    for result in client.results(search):
        papers.append(
            {
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "published": result.published.strftime("%Y-%m-%d"),
                "summary": result.summary,
                "pdf_url": result.pdf_url,
            }
        )
    return papers

arxiv_researcher_agent = AssistantAgent(
    name='arxiv_search_agent',
    description='Create arXiv queries and retrieves candidates papers',
    model_client=autogen_brain,
    tools=[arxiv_search],
    system_message=(
            "Given a user topic, think of the best arXiv query. When the tool "
            "returns, choose exactly the number of papers requested and pass "
            "them as concise JSON to the summarizer."
        ),
)

summarizer_agent = AssistantAgent(
    name='summarizer_agent',
    description = 'An agent which summarizes the result',
    model_client=autogen_brain,
    system_message=(
            "You are an expert researcher. When you receive the JSON list of "
            "papers, write a literature‑review style report in Markdown:\n" \
            "1. Start with a 2–3 sentence introduction of the topic.\n" \
            "2. Then include one bullet per paper with: title (as Markdown "
            "link), authors, the specific problem tackled, and its key "
            "contribution.\n" \
            "3. Close with a single‑sentence takeaway."
        ),
)

team = RoundRobinGroupChat(
    participants=[arxiv_researcher_agent,summarizer_agent],
    max_turns=2
)

async def run_team():

    task = 'Condict a literature review on the topic - Autogen and return exactly 5 papers.'
    async for msg in team.run_stream(task=task):
        print(msg)

if (__name__=='__main__'):
    asyncio.run(run_team())