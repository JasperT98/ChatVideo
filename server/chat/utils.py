from pydantic import BaseModel, Field
from langchain.prompts.chat import ChatMessagePromptTemplate
from configs import logger, log_verbose
from typing import List, Tuple, Dict, Union


class History(BaseModel):
    """
    对话历史
    可从dict生成，如
    h = History(**{"role":"user","content":"你好"})
    也可转换为tuple，如
    h.to_msy_tuple = ("human", "你好")
    """
    role: str = Field(...)
    content: str = Field(...)

    def to_msg_tuple(self):
        return "ai" if self.role=="assistant" else "human", self.content

    def to_msg_template(self, is_raw=True) -> ChatMessagePromptTemplate:
        role_maps = {
            "ai": "assistant",
            "human": "user",
        }
        role = role_maps.get(self.role, self.role)
        if is_raw: # 当前默认历史消息都是没有input_variable的文本。
            content = "{% raw %}" + self.content + "{% endraw %}"
        else:
            content = self.content

        # `ChatMessagePromptTemplate.from_template()` 是一个函数，它接受三个参数：`content`，`"jinja2"` 和 `role`。
        # `content` 是一个字符串，表示聊天消息的模板。
        # `"jinja2"` 是一个字符串，表示模板的格式。
        # `role` 是一个字符串，表示消息的角色。
        # 该函数返回一个 `ChatMessagePromptTemplate` 对象，该对象包含了聊天消息的模板和角色信息。
        return ChatMessagePromptTemplate.from_template(
            content,
            "jinja2",
            role=role,
        )

    @classmethod
    def from_data(cls, h: Union[List, Tuple, Dict]) -> "History":
        if isinstance(h, (list,tuple)) and len(h) >= 2:
            h = cls(role=h[0], content=h[1])
        elif isinstance(h, dict):
            h = cls(**h)

        return h
