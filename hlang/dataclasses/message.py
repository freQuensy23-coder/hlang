from enum import Enum
from typing import Literal, Optional


class ChatRole(str, Enum):
    """Enumeration representing the roles within a chat."""
    ASSISTANT = "assistant"
    USER = "user"
    SYSTEM = "system"
    FUNCTION = "function"


class ChatMessage:
    def __init__(self, content: str, role: ChatRole, function_name: Optional[str] = None):
        self.content = content
        self.role = role
        self.function_name = function_name

    @classmethod
    def from_user(cls, content: str):
        return cls(content, role=ChatRole('user'))

    @classmethod
    def from_assistant(cls, content: str):
        return cls(content, role=ChatRole('assistant'))

    @classmethod
    def from_system(cls, content: str):
        return cls(content, role=ChatRole('system'))

    def to_dict(self):
        if self.function_name:
            return {
                "content": self.content,
                "role": self.role,
                "function_name": self.function_name
            }
        return {
            "content": self.content,
            "role": self.role
        }
