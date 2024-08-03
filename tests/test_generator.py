from unittest import TestCase
from unittest.mock import patch, MagicMock

from hlang.dataclasses.message import ChatMessage
from hlang.generators.openai_generator import OpenAIChatGenerator


class TestOpenaiGenerator(TestCase):
    def setUp(self):
        self.model_name = "test-model"
        self.base_url = "http://test-url.com"
        self.generator = OpenAIChatGenerator(model_name=self.model_name, base_url=self.base_url)

    def test_generate_single_message(self):
        with patch('hlang.openai_lazy_client.OpenAILazyClient') as MockClient:
            mock_client_instance = MockClient.return_value
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
            mock_client_instance.chat.completions.create.return_value = mock_response

            messages = [ChatMessage.from_user("Hello")]
            result = self.generator.generate(messages=messages)

            self.assertEqual(result.content, "Test response")
