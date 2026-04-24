from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from utils.config_handler import rag_conf
import os  # 必须加！


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        # 修复：从环境变量读取 API Key，支持Streamlit线上部署
        return ChatTongyi(
            model=rag_conf["chat_model_name"],
            dashscope_api_key=os.getenv("sk-4ca7427ae3e24a3096227ab8b6191233")  # 关键修复
        )


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        # 修复：嵌入模型也必须读API Key
        return DashScopeEmbeddings(
            model=rag_conf["embedding_model_name"],
            dashscope_api_key=os.getenv("sk-4ca7427ae3e24a3096227ab8b6191233")  # 关键修复
        )


chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()