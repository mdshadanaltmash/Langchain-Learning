from dotenv import load_dotenv
from pprint import pprint
from graph.graph import app


load_dotenv()


if __name__ == '__main__':
    print("Hello Advanced Agentic RAG!")
    # pprint(app.invoke(input={"question": "What is agent memory?"}))
    pprint(app.invoke(input={"question": "Who is Md Shadan Altmash?"}))
