from jinja2 import Template

from hlang.dataclasses.message import ChatMessage


class ChatPromptBuilder:
    def __init__(self, prompt_template: [ChatMessage]):
        self.prompt_template = prompt_template
        self.template = [Template(m.content) for m in prompt_template]

    def run(self, **kwargs):
        rendered_content = [t.render(**kwargs) for t in self.template]
        result = []
        for m in self.prompt_template:
            result.append(ChatMessage(content=rendered_content.pop(0), role=m.role))
        return result

