import warnings
from abc import ABC, abstractmethod

from hlang.dataclasses.message import ChatMessage, ChatRole


class ABCGenerator(ABC):
    @abstractmethod
    def generate(self, *args, **kwargs):
        pass


class ABCTextGenerator(ABCGenerator):
    @abstractmethod
    def generate(self, text: str) -> str:
        pass


class ABCChatGenerator(ABCGenerator):
    @abstractmethod
    def generate(self, messages: [ChatMessage]) -> ChatMessage:
        pass

    @staticmethod
    def _prompt_sanity_check(prompt):
        if prompt[0].role not in {ChatRole.USER, ChatRole.SYSTEM}:
            warnings.warn("The first message in the prompt should be from the user or the system.")
        if prompt[-1].role == ChatRole.ASSISTANT:
            warnings.warn("The last message in the prompt should not be from the assistant. Can't generate a response "
                          "to the assistant.")



