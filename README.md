# hlang Library

`hlang` is a Python library designed to facilitate the creation and management of various AI and natural language processing (NLP) tasks. It provides modules for embedding documents, generating responses using different models, and constructing chat prompts. The library also includes a vector database for efficient document retrieval.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Document Handling](#document-handling)
  - [Embedding Text and Documents](#embedding-text-and-documents)
  - [Vector Database](#vector-database)
  - [Prompt Building](#prompt-building)
  - [Response Generation](#response-generation)
- [Examples](#examples)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Installation

To install `hlang`, clone the repository and install the required dependencies:

```bash
pip install -U hlang
```

## Usage

### Document Handling

`hlang` provides classes to handle different types of documents. The `Document` class is a simple document container, while the `KeyValueDocument` allows for more structured data storage.

```python
from hlang.documents.simple_document import Document
from hlang.documents.key_value_document import KeyValueDocument

doc = Document(content="Hello, world!")
kv_doc = KeyValueDocument(key="description", content="A simple document")
```

### Embedding Text and Documents

Embedding functionality is provided by the `SentenceTransformerEmbedder` class, which uses the `sentence-transformers` library.

```python
from hlang.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder

embedder = SentenceTransformerEmbedder(model_name="intfloat/e5-small-v2")
text_embedding = embedder.embed_text(["Hello, world!"])
```

### Vector Database

The `VectorStorage` class is used to store and search documents using their embeddings. It leverages the `chromadb` client for efficient vector operations.

```python
import torch
from hlang.vectordb.vector_db import VectorStorage
from hlang.documents.simple_document import Document

documents = [Document(f"Document {i}") for i in range(10)]
embeddings = torch.rand(10, 312)

db = VectorStorage(documents, embeddings)
query = torch.rand(1, 312)
results = db.search(query)
```

### Prompt Building

The `ChatPromptBuilder` class allows for dynamic construction of chat prompts using Jinja2 templates.

```python
from hlang.dataclasses.message import ChatMessage, ChatRole
from hlang.templates.chat_prompt_template import ChatPromptBuilder

prompt_template = [
    ChatMessage.from_user("Hello, {{ name }}!"),
    ChatMessage.from_assistant("Your age is {{ age }}.")
]
builder = ChatPromptBuilder(prompt_template)
rendered_messages = builder.run(name="Alice", age=30)
```

### Response Generation

`hlang` supports response generation using OpenAI's GPT models via the `OpenAIChatGenerator` class.

```python
from hlang.dataclasses.message import ChatMessage
from hlang.generators.openai_generator import OpenAIChatGenerator

llm = OpenAIChatGenerator(model_name="gpt-3.5-turbo")
messages = [ChatMessage.from_user("Tell me a joke")]
response = llm.generate(messages)
print(response.content)
```

## Examples

The `examples` directory contains various usage examples, such as `rag_openai.py`, which demonstrates a retrieval-augmented generation pipeline.

```bash
python examples/rag_openai.py
```

## Testing

Unit tests are located in the `tests` directory. To run the tests, use the following command:

```bash
pytest tests
```

## Contributing

Contributions are welcome! Please read the [contribution guidelines](CONTRIBUTING.md) (% TODO %) and adhere to the project's coding standards.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

By using `hlang`, you can streamline the development of NLP applications, leveraging state-of-the-art models and efficient data handling practices.