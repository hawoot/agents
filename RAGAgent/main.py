from dotenv import load_dotenv
load_dotenv()

from graph.state import GraphState

state = GraphState(question="Who does O'Brien represent as a historical figure? In the context of the 20th century? what about the 21st century?", generation="L6", web_search=True, documents=[])
from graph.nodes.retrieve import retrieve
result = retrieve(state)
print(result['documents'])