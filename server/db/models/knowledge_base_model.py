from sqlalchemy import Column, Integer, String, DateTime, func

from server.db.base import Base


class KnowledgeBaseModel(Base):
    """
    知识库模型
    """
    __tablename__ = 'knowledge_base'
    id = Column(Integer, primary_key=True, autoincrement=True, comment='知识库ID')
    kb_name = Column(String(50), comment='知识库名称')
    vs_type = Column(String(50), comment='向量库类型')
    embed_model = Column(String(50), comment='嵌入模型名称')
    file_count = Column(Integer, default=0, comment='文件数量')
    create_time = Column(DateTime, default=func.now(), comment='创建时间')
    video_path = Column(String(255), comment='视频路径')

    def __repr__(self):
        return (f"<KnowledgeBase(id='{self.id}', "
                f"kb_name='{self.kb_name}', "
                f"vs_type='{self.vs_type}', "
                f"embed_model='{self.embed_model}', "
                f"file_count='{self.file_count}', "
                f"video_path='{self.video_path}', "
                f"create_time='{self.create_time}')>")
