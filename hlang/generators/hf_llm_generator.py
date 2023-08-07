import transformers


class HuggingfaceLLMGenerator:
    def __init__(self, model_name: str):
        self.model = transformers.AutoModel.from_pretrained(model_name)
        self.tokenizer = transformers.AutoModel.from_pretrained(model_name)

    def generate(self, prompt: str, **kwargs) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        output = self.model.generate(input_ids, **kwargs)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
