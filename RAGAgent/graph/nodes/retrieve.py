from ..state import GraphState

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_pinecone import PineconeVectorStore

db_provider = "pinecone"
embeddings_model = "sentence-transformers/all-MiniLM-L6-v2"
embedding = SentenceTransformerEmbeddings(model_name=embeddings_model)
vectorstore = PineconeVectorStore(embedding=embedding, index_name="orwell-1984")
# vectorstore.as_retriever().get_relevant_documents("Who does O'Brien represent as a historical figure? In the context of the 20th century? what about the 21st century?")

def retrieve(state: GraphState) -> GraphState:
    """
    Retrieve documents relevant to the question
    
    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """
    print("---RETRIEVE---")
    question = state['question']
    print(f"Running embeddings model: {embeddings_model}. Vectorstore provider: {db_provider}")
    documents = vectorstore.as_retriever().get_relevant_documents(question)
    return {'documents': documents}