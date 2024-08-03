from unittest import TestCase

from openlangchain.dataclasses.message import ChatRole, ChatMessage


class TestChatMessage(TestCase):

    def test_initialization(self):
        message = ChatMessage("Hello", ChatRole.USER)
        self.assertEqual(message.content, "Hello")
        self.assertEqual(message.role, ChatRole.USER)
        self.assertIsNone(message.function_name)

    def test_initialization_with_function(self):
        message = ChatMessage("Hello", ChatRole.FUNCTION, "some_function")
        self.assertEqual(message.content, "Hello")
        self.assertEqual(message.role, ChatRole.FUNCTION)
        self.assertEqual(message.function_name, "some_function")

    def test_from_user(self):
        message = ChatMessage.from_user("Hi there")
        self.assertEqual(message.content, "Hi there")
        self.assertEqual(message.role, ChatRole.USER)
        self.assertIsNone(message.function_name)

    def test_from_assistant(self):
        message = ChatMessage.from_assistant("Hello, how can I help?")
        self.assertEqual(message.content, "Hello, how can I help?")
        self.assertEqual(message.role, ChatRole.ASSISTANT)
        self.assertIsNone(message.function_name)

    def test_from_system(self):
        message = ChatMessage.from_system("System update")
        self.assertEqual(message.content, "System update")
        self.assertEqual(message.role, ChatRole.SYSTEM)
        self.assertIsNone(message.function_name)