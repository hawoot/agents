from dotenv import load_dotenv
load_dotenv()

from  graph.graph import compile_graph

if __name__ == "__main__":
    app = compile_graph()
    # question = "What is the capital of France?"
    question="Who does O'Brien represent as a historical figure? In the context of the 20th century? what about the 21st century?"
    result = app.invoke({"question": question})

    print("#"*20)
    for key, value in result.items():
        print(key)
        print(value)
        print("#"*20)