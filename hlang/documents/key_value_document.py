from hlang.documents.document import ABCDocument


class KeyValueDocument(ABCDocument):
    def __init__(self, content, key: str):
        self.key = key
        self.value = content

    def embed_text(self) -> str:
        return self.key
