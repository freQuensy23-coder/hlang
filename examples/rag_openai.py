from hlang.dataclasses.message import ChatMessage
from hlang.documents.simple_document import Document
from hlang.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
from hlang.generators.openai_generator import OpenAIChatGenerator
from hlang.templates.chat_prompt_template import ChatPromptBuilder
from hlang.vectordb.vector_db import VectorStorage

knowledge_base = [
    "Joe lives in Berlin",
    "Joe is a software engineer",
    "Joe likes to play football",
    "Joe is married to Maria",
    "Joe has two children",
    "Joe likes to travel",
    "Joe is a vegetarian",
    "Joe is a good cook",
    "Joe was born in 1980",
    "Joe is a fan of Star Wars",
    "Joe is a fan of The Beatles",
]
prompt = [
    ChatMessage.from_system("Help user to answer question based on information "
                            "{% for doc in documents %}{{ doc.content }} \n{% endfor %}"),
    ChatMessage.from_user('Please answer my question based on the information and do not provide any new '
                          'information. My question {{ question }}'),
]
prompt_builder = ChatPromptBuilder(prompt)

documents = [Document(content=doc) for doc in knowledge_base]

embedder = SentenceTransformerEmbedder(model_name="intfloat/e5-small-v2")
embeddings = embedder.embed_documents(documents)
db = VectorStorage(documents, embeddings)

llm = OpenAIChatGenerator(model_name="gpt4o-mini")


def process(user_query: str):
    query_embedding = embedder.embed_text([user_query])[0]
    retrival_docs = db.search(query_embedding, top_k=2)

    result = llm.generate(
        messages=prompt_builder.run(question=question, documents=retrival_docs)
    )

    print(result.content)


while True:
    question = input("Ask a question: ")
    process(question)
