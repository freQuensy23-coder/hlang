from unittest import TestCase

from hlang.dataclasses.message import ChatMessage, ChatRole
from hlang.templates.chat_prompt_template import ChatPromptBuilder


class TestChatPromptBuilder(TestCase):

    def test_single_variable(self):
        prompt_template = [ChatMessage.from_user("Hello, {{ name }}!")]
        builder = ChatPromptBuilder(prompt_template)
        rendered_messages = builder.run(name="World")
        self.assertEqual(len(rendered_messages), 1)
        self.assertEqual(rendered_messages[0].content, "Hello, World!")
        self.assertEqual(rendered_messages[0].role, ChatRole.USER)

    def test_multiple_variables(self):
        prompt_template = [
            ChatMessage.from_user("Hello, {{ name }}!"),
            ChatMessage.from_assistant("Your age is {{ age }}.")
        ]
        builder = ChatPromptBuilder(prompt_template)
        rendered_messages = builder.run(name="Alice", age=30)
        self.assertEqual(len(rendered_messages), 2)
        self.assertEqual(rendered_messages[0].content, "Hello, Alice!")
        self.assertEqual(rendered_messages[0].role, ChatRole.USER)
        self.assertEqual(rendered_messages[1].content, "Your age is 30.")
        self.assertEqual(rendered_messages[1].role, ChatRole.ASSISTANT)

    def test_no_variables(self):
        prompt_template = [
            ChatMessage.from_user("Hello!"),
            ChatMessage.from_assistant("How can I help you?")
        ]
        builder = ChatPromptBuilder(prompt_template)
        rendered_messages = builder.run()
        self.assertEqual(len(rendered_messages), 2)
        self.assertEqual(rendered_messages[0].content, "Hello!")
        self.assertEqual(rendered_messages[0].role, ChatRole.USER)
        self.assertEqual(rendered_messages[1].content, "How can I help you?")
        self.assertEqual(rendered_messages[1].role, ChatRole.ASSISTANT)
