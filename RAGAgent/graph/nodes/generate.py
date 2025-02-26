from langchain import hub
from langchain_core.output_parsers import StrOutputParser

# from langchain_openai import ChatOpenAI
# model, provider = "gpt-4o-mini", "openai"
# llm = ChatOpenAI(temperature=0, model=model')
# llm_details = {'llm': llm, 'model': model, 'provider': provider}
from langchain_groq import ChatGroq
provider = "groq"
model  = "deepseek-r1-distill-llama-70b"
# model = "mixtral-8x7b-32768"
# model = "qwen-2.5-32b"
# model = "mistral-saba-24b"
llm = ChatGroq(temperature=0, model_name=model)
llm_details = {'llm': llm, 'model': model, 'provider': provider}


prompt = hub.pull("rlm/rag-prompt")
generator = prompt | llm_details['llm'] | StrOutputParser()

from typing import Any, Dict
from ..state import GraphState

def generate(state: GraphState) -> Dict[str, Any]:
    """
    Generates a response to the question using the retrieved documents
    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    print(f"Running model: {llm_details['model']} from {llm_details['provider']}")
    generation = generator.invoke({"question": question, "context": documents})
    return {"generation": generation}