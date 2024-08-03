from hlang.documents.document import ABCDocument


class Document(ABCDocument):
    def __init__(self, content: str):
        self.content = content

    def embed_text(self) -> str:
        return self.content
