import os

from openai import Client


class OpenAILazyClient(Client):
    def __init__(self, **kwargs):
        super().__init__(api_key='sk-1234567890abcdef1234567890abcdef',
                         **kwargs)

    def set_key(self):
        self.api_key = os.getenv('OPENAI_API_KEY')

    def __getattribute__(self, name):
        # Intercept attribute access
        if name != 'set_key' and callable(super().__getattribute__(name)):
            # Call set_key before any method other than set_key itself
            self.set_key()
        return super().__getattribute__(name)
