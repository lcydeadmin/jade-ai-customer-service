from abc import ABC, abstractmethod
from typing import Optional

# 1. 导入修正：ChatTongyi 位于 chat_models.tongyi
from langchain_community.chat_models.tongyi import ChatTongyi

# 2. 导入修正：BaseChatModel 位于 core 包中，不要从 community 里导
from langchain_core.chat_models import BaseChatModel

from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import DashScopeEmbeddings
from utils.config_handler import rag_conf
import os

class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        # 修复：从环境变量读取 API Key，支持Streamlit线上部署
        return ChatTongyi(
            model=rag_conf["chat_model_name"],
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")  # 关键修复
        )


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        # 修复：嵌入模型也必须读API Key
        return DashScopeEmbeddings(
            model=rag_conf["embedding_model_name"],
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")  # 关键修复
        )


chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()