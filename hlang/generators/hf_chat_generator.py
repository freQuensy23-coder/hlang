import warnings

import transformers
from transformers import PreTrainedModel

from hlang.dataclasses.message import ChatMessage
from hlang.generators.generator import ABCChatGenerator


class HuggingfaceChatLLMGenerator(ABCChatGenerator):
    def __init__(self, model_name: str):
        self.model: PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = transformers.AutoModel.from_pretrained(model_name)

    def generate(self, prompt: [ChatMessage], **kwargs):
        self._prompt_sanity_check(prompt)

        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        output = self.model.generate(input_ids, **kwargs)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
