from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# from langchain_openai import ChatOpenAI
# model, provider = "gpt-4o-mini", "openai"
# llm = ChatOpenAI(temperature=0, model=model')
# llm_details = {'llm': llm, 'model': model, 'provider': provider}
from langchain_groq import ChatGroq
provider = "groq"
# model  = "deepseek-r1-distill-llama-70b",
# model = "mixtral-8x7b-32768"
# model = "qwen-2.5-32b"
model = "mistral-saba-24b"
llm = ChatGroq(temperature=0, model_name=model)
llm_details = {'llm': llm, 'model': model, 'provider': provider}

class GradeDocument(BaseModel):
    binary_score: str = Field(description="Document is relevant to the question, 'yes' or 'no'")
structured_llm_grader = llm_details['llm'].with_structured_output(GradeDocument)

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ("human", "I am collecting documents that might be related to the question, it doesn't have to answer the question fully"),
        ("human", "Tell me whether it is relevant to the question, 'yes' or 'no'"),
    ]
)
grader = grade_prompt | structured_llm_grader


from typing import Any, Dict
from ..state import GraphState

def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False
    for d in documents:
        print(f"Running model: {llm_details['model']} from {llm_details['provider']}")
        score = grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = True
            continue
    return {"documents": filtered_docs, "web_search": web_search}