import os
import urllib
from urllib.parse import urlparse

from server.utils import BaseResponse, ListResponse
from server.knowledge_base.utils import validate_kb_name
from server.knowledge_base.kb_service.base import KBServiceFactory
from server.db.repository.knowledge_base_repository import list_kbs_from_db, get_kb_detail
from configs import EMBEDDING_MODEL, logger, log_verbose
from fastapi import Body


def is_url_or_path(path):
    # 检查是否为有效的网址
    try:
        result = urlparse(path)
        if all([result.scheme, result.netloc]):
            return True
    except:
        pass

    # 检查是否为有效的文件系统路径
    if os.path.exists(path):
        return True

    return False


def list_video_path(
        knowledge_base_name: str,
) -> ListResponse:
    logger.info(get_kb_detail(kb_name=knowledge_base_name))
    try:
        if not validate_kb_name(knowledge_base_name):
            return ListResponse(code=403, msg="Don't attack me", data=[])
        kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
        if kb is None:
            return ListResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}", data=[])
        video_path = kb.get_video_path_in_kb()
        if video_path is None:
            return ListResponse(code=200, msg=f"知识库{knowledge_base_name}未找到视频地址", data=[])
        return ListResponse(code=200, data=[video_path])
    except Exception as e:
        msg = f"获取视频路径出错： {e}"
        return ListResponse(code=500, msg=f'{e.__class__.__name__}: {msg}', data=[])


# 尽量不要开放，防止调用api乱改
def update_video_path(
        knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
        video_path: str = Body(None, description="视频路径", examples=["kd/xxx/xxx.mp4", "youtube.com/watch?v=xxx"])
) -> BaseResponse:
    # 调用时不给video_path附值会把video_path设为None（对应数据库的NULL）
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    # 拿到kb_service里的某个kb服务实例，比如：faiss_kb_service, ...（都继承kb_service（base文件下））
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")
    # 检查是否为有效的网址或文件系统路径
    if video_path is not None and is_url_or_path(video_path):
        return BaseResponse(code=400, msg=f"视频路径无效")
    try:
        kb.update_video_path_in_kb(video_path)
    except Exception as e:
        msg = f"更新视频路径出错： {e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
        return BaseResponse(code=500, msg=msg)
    return BaseResponse(code=200, msg=f"已更新视频路径")


def list_kbs():
    # Get List of Knowledge Base
    return ListResponse(data=list_kbs_from_db())


def create_kb(knowledge_base_name: str = Body(..., examples=["samples"]),
              vector_store_type: str = Body("faiss"),
              embed_model: str = Body(EMBEDDING_MODEL),
              ) -> BaseResponse:
    # Create selected knowledge base
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")
    if knowledge_base_name is None or knowledge_base_name.strip() == "":
        return BaseResponse(code=404, msg="知识库名称不能为空，请重新填写知识库名称")

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is not None:
        return BaseResponse(code=404, msg=f"已存在同名知识库 {knowledge_base_name}")

    kb = KBServiceFactory.get_service(knowledge_base_name, vector_store_type, embed_model)
    try:
        kb.create_kb()
    except Exception as e:
        msg = f"创建知识库出错： {e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
        return BaseResponse(code=500, msg=msg)

    return BaseResponse(code=200, msg=f"已新增知识库 {knowledge_base_name}")


def delete_kb(
        knowledge_base_name: str = Body(..., examples=["samples"])
) -> BaseResponse:
    # Delete selected knowledge base
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")
    knowledge_base_name = urllib.parse.unquote(knowledge_base_name)

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)

    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    try:
        status = kb.clear_vs()
        status = kb.drop_kb()
        if status:
            return BaseResponse(code=200, msg=f"成功删除知识库 {knowledge_base_name}")
    except Exception as e:
        msg = f"删除知识库时出现意外： {e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
        return BaseResponse(code=500, msg=msg)

    return BaseResponse(code=500, msg=f"删除知识库失败 {knowledge_base_name}")
