
# ✅ 修改后的正确写法
from langchain_core.embeddings import Embeddings

# 注意这里变成了 llms 而不是 chat_models
from langchain_community.llms.tongyi import ChatTongyi

# 如果你需要 BaseChatModel，它通常在 core 或者 llms 里，
# 但 ChatTongyi 继承自 BaseChatModel，通常直接导入 ChatTongyi 就够了。
# 如果确实需要基类，建议检查是否应为：
# from langchain_core.language_models.chat_models import BaseChatModel

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