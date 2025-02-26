# !pip install langchain-community duckduckgo-search

from langchain.schema import Document
import json

# from langchain_community.tools import DuckDuckGoSearchResults
# provider = "DuckDuckGo"
# web_search_tool = DuckDuckGoSearchResults(output_format="json")

from langchain_community.tools.tavily_search import TavilySearchResults
provider = "tavily"
web_search_tool = TavilySearchResults(k=3)

from typing import Any, Dict
from langchain.schema import Document
from ..state import GraphState

def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    print(f"Running web search tool: {provider}")
    docs = web_search_tool.invoke({"query": question})
    # # DuckDuckGo
    # web_results = '\n\n'.join('\n'.join(f"{key}: {value}" for key, value in doc.items()) for doc in json.loads(docs))
    # Tavily
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents}