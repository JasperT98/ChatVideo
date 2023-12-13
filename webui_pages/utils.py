# 该文件包含webui通用工具，可以被不同的webui使用
import functools
from functools import wraps
from typing import *
from pathlib import Path
from configs import (
    EMBEDDING_MODEL,
    DEFAULT_VS_TYPE,
    KB_ROOT_PATH,
    LLM_MODEL,
    HISTORY_LEN,
    TEMPERATURE,
    SCORE_THRESHOLD,
    CHUNK_SIZE,
    OVERLAP_SIZE,
    ZH_TITLE_ENHANCE,
    VECTOR_SEARCH_TOP_K,
    SEARCH_ENGINE_TOP_K,
    FSCHAT_MODEL_WORKERS,
    HTTPX_DEFAULT_TIMEOUT,
    logger, log_verbose,
)
import httpx
import asyncio
from server.chat.openai_chat import OpenAiChatMsgIn
from fastapi.responses import StreamingResponse
import contextlib
import json
import os
from io import BytesIO
from server.utils import run_async, iter_over_async, set_httpx_config, api_address, get_httpx_client

from configs.model_config import NLTK_DATA_PATH
import nltk

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path
from pprint import pprint

KB_ROOT_PATH = Path(KB_ROOT_PATH)
set_httpx_config()


async def async_run_multi_func(*args, done_callback_func=lambda f: logger.warning(
    f'The result of the gather is {f.result()}')):
    '''
    函数接受任意数量的参数。
    使用 asyncio.gather() 函数来并发运行所有传入的参数。
    asyncio.gather 函数会等待所有的异步任务完成后返回结果。
    这个函数的返回值是一个 Future 对象，可以使用 await 关键字来获取结果。
    '''
    task_list = []  # 任务列表
    for func in args:
        task = asyncio.create_task(func)  # 创建任务
        task_list.append(task)
    return await asyncio.gather(*task_list)

    # gathered = asyncio.gather(*args)
    # # Add a callback to monitor the future object
    # gathered.add_done_callback(done_callback_func)
    # # Wait for the gather to complete
    # results = await gathered
    # return results


async def async_decorator(func):
    '''
    异步函数装饰器：将一个非异步函数变成异步函数
    '''

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)

    return wrapper


def convert_file(file, filename=None):
    if isinstance(file, bytes):  # raw bytes
        # BytesIO 是 Python 的 io 模块中的一个类，它提供了一种处理内存中二进制数据的方法，就像处理文件一样。
        # 它允许您从内存缓冲区中读取和写入数据，这在需要操作二进制数据但不想将其写入磁盘的情况下非常有用。
        file = BytesIO(file)
        logger.info(" ================ raw bytes ======================")
    elif hasattr(file, "read"):  # a file io like object
        # 用于检查 file 是否具有 read 属性。
        # 如果 file 具有 read 属性，则说明它是一个文件 I/O 类似对象。
        # 在这种情况下，filename 变量将设置为 filename 或 file.name，具体取决于哪个值存在且不为 None。
        filename = filename or file.name
        logger.info(" ================ a file io like object ======================")
        logger.info(filename)
    else:  # a local path
        # 读取本地文件
        file = Path(file).absolute().open("rb")
        filename = filename or os.path.split(file.name)[-1]
        logger.info(" ================ a local path ======================")
        logger.info(filename)
    return filename, file


class ApiRequest:
    '''
    api.py调用的封装,主要实现:
    1. 简化api调用方式
    2. 实现无api调用(直接运行server.chat.*中的视图函数获取结果),无需启动api.py
    '''

    def __init__(
            self,
            base_url: str = api_address(),
            timeout: float = HTTPX_DEFAULT_TIMEOUT,
            no_remote_api: bool = False,  # call api view function directly
    ):
        self.base_url = base_url
        self.timeout = timeout
        self.no_remote_api = no_remote_api
        self._client = get_httpx_client()
        self._aclient = get_httpx_client(use_async=True)
        if no_remote_api:
            logger.warn("将来可能取消对no_remote_api的支持，更新版本时请注意。")

    def _parse_url(self, url: str) -> str:
        if (not url.startswith("http")
                and self.base_url
        ):
            part1 = self.base_url.strip(" /")
            part2 = url.strip(" /")
            return f"{part1}/{part2}"
        else:
            return url

    def get(
            self,
            url: str,
            params: Union[Dict, List[Tuple], bytes] = None,
            retry: int = 3,
            stream: bool = False,
            **kwargs: Any,
    ) -> Union[httpx.Response, None]:
        url = self._parse_url(url)
        kwargs.setdefault("timeout", self.timeout)
        while retry > 0:
            try:
                if stream:
                    return self._client.stream("GET", url, params=params, **kwargs)
                else:
                    return self._client.get(url, params=params, **kwargs)
            except Exception as e:
                msg = f"error when get {url}: {e}"
                logger.error(f'{e.__class__.__name__}: {msg}',
                             exc_info=e if log_verbose else None)
                retry -= 1

    async def aget(
            self,
            url: str,
            params: Union[Dict, List[Tuple], bytes] = None,
            retry: int = 3,
            stream: bool = False,
            **kwargs: Any,
    ) -> Union[httpx.Response, None]:
        url = self._parse_url(url)
        kwargs.setdefault("timeout", self.timeout)

        while retry > 0:
            try:
                if stream:
                    return await self._aclient.stream("GET", url, params=params, **kwargs)
                else:
                    return await self._aclient.get(url, params=params, **kwargs)
            except Exception as e:
                msg = f"error when aget {url}: {e}"
                logger.error(f'{e.__class__.__name__}: {msg}',
                             exc_info=e if log_verbose else None)
                retry -= 1

    def post(
            self,
            url: str,
            data: Dict = None,
            json: Dict = None,
            retry: int = 3,
            stream: bool = False,
            **kwargs: Any
    ) -> Union[httpx.Response, None]:
        url = self._parse_url(url)
        kwargs.setdefault("timeout", self.timeout)
        while retry > 0:
            try:
                if stream:
                    return self._client.stream("POST", url, data=data, json=json, **kwargs)
                else:
                    return self._client.post(url, data=data, json=json, **kwargs)
            except Exception as e:
                msg = f"error when post {url}: {e}"
                logger.error(f'{e.__class__.__name__}: {msg}',
                             exc_info=e if log_verbose else None)
                retry -= 1

    async def apost(
            self,
            url: str,
            data: Dict = None,
            json: Dict = None,
            retry: int = 3,
            stream: bool = False,
            **kwargs: Any
    ) -> Union[httpx.Response, None]:
        url = self._parse_url(url)
        kwargs.setdefault("timeout", self.timeout)

        while retry > 0:
            try:
                if stream:
                    res = await self._client.stream("POST", url, data=data, json=json, **kwargs)
                    logger.info("=================== upload done =================")
                    return res
                else:
                    res = await self._client.post(url, data=data, json=json, **kwargs)
                    logger.info("=================== upload done =================")
                    return res
            except Exception as e:
                msg = f"error when apost {url}: {e}"
                logger.error(f'{e.__class__.__name__}: {msg}',
                             exc_info=e if log_verbose else None)
                retry -= 1

    def delete(
            self,
            url: str,
            data: Dict = None,
            json: Dict = None,
            retry: int = 3,
            stream: bool = False,
            **kwargs: Any
    ) -> Union[httpx.Response, None]:
        url = self._parse_url(url)
        kwargs.setdefault("timeout", self.timeout)
        while retry > 0:
            try:
                if stream:
                    return self._client.stream("DELETE", url, data=data, json=json, **kwargs)
                else:
                    return self._client.delete(url, data=data, json=json, **kwargs)
            except Exception as e:
                msg = f"error when delete {url}: {e}"
                logger.error(f'{e.__class__.__name__}: {msg}',
                             exc_info=e if log_verbose else None)
                retry -= 1

    async def adelete(
            self,
            url: str,
            data: Dict = None,
            json: Dict = None,
            retry: int = 3,
            stream: bool = False,
            **kwargs: Any
    ) -> Union[httpx.Response, None]:
        url = self._parse_url(url)
        kwargs.setdefault("timeout", self.timeout)

        while retry > 0:
            try:
                if stream:
                    return await self._aclient.stream("DELETE", url, data=data, json=json, **kwargs)
                else:
                    return await self._aclient.delete(url, data=data, json=json, **kwargs)
            except Exception as e:
                msg = f"error when adelete {url}: {e}"
                logger.error(f'{e.__class__.__name__}: {msg}',
                             exc_info=e if log_verbose else None)
                retry -= 1

    def _fastapi_stream2generator(self, response: StreamingResponse, as_json: bool = False):
        '''
        将api.py中视图函数返回的StreamingResponse转化为同步生成器
        '''
        try:
            loop = asyncio.get_event_loop()
        except:
            loop = asyncio.new_event_loop()

        try:
            for chunk in iter_over_async(response.body_iterator, loop):
                if as_json and chunk:
                    yield json.loads(chunk)
                elif chunk.strip():
                    yield chunk
        except Exception as e:
            msg = f"error when run fastapi router: {e}"
            logger.error(f'{e.__class__.__name__}: {msg}',
                         exc_info=e if log_verbose else None)

    def _httpx_stream2generator(
            self,
            response: contextlib._GeneratorContextManager,
            as_json: bool = False,
    ):
        '''
        将httpx.stream返回的GeneratorContextManager转化为普通生成器
        '''
        try:
            with response as r:
                for chunk in r.iter_text(None):
                    if not chunk:  # fastchat api yield empty bytes on start and end
                        continue
                    if as_json:
                        try:
                            logger.error(chunk)
                            data = json.loads(chunk)
                            pprint(data, depth=1)
                            yield data
                        except Exception as e:
                            msg = f"接口返回json错误： ‘{chunk}’。错误信息是：{e}。"
                            logger.error(f'{e.__class__.__name__}: {msg}',
                                         exc_info=e if log_verbose else None)
                    else:
                        print(chunk, end="", flush=True)
                        yield chunk
        except httpx.ConnectError as e:
            msg = f"无法连接API服务器，请确认 ‘api.py’ 已正常启动。({e})"
            logger.error(msg)
            logger.error(msg)
            yield {"code": 500, "msg": msg}
        except httpx.ReadTimeout as e:
            msg = f"API通信超时，请确认已启动FastChat与API服务（详见RADME '5. 启动 API 服务或 Web UI'）。（{e}）"
            logger.error(msg)
            yield {"code": 500, "msg": msg}
        except Exception as e:
            msg = f"API通信遇到错误：{e}"
            logger.error(f'{e.__class__.__name__}: {msg}',
                         exc_info=e if log_verbose else None)
            yield {"code": 500, "msg": msg}

    # 对话相关操作

    def chat_fastchat(
            self,
            messages: List[Dict],
            stream: bool = True,
            model: str = LLM_MODEL,
            temperature: float = TEMPERATURE,
            max_tokens: int = 1024,  # todo:根据message内容自动计算max_tokens
            no_remote_api: bool = None,
            **kwargs: Any,
    ):
        '''
        对应api.py/chat/fastchat接口
        '''
        if no_remote_api is None:
            no_remote_api = self.no_remote_api
        msg = OpenAiChatMsgIn(**{
            "messages": messages,
            "stream": stream,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        })

        if no_remote_api:
            from server.chat.openai_chat import openai_chat
            response = run_async(openai_chat(msg))
            return self._fastapi_stream2generator(response)
        else:
            data = msg.dict(exclude_unset=True, exclude_none=True)
            print(f"received input message:")
            pprint(data)

            response = self.post(
                "/chat/fastchat",
                json=data,
                stream=True,
            )
            return self._httpx_stream2generator(response)

    def chat_chat(
            self,
            query: str,
            history: List[Dict] = [],
            stream: bool = True,
            model: str = LLM_MODEL,
            temperature: float = TEMPERATURE,
            prompt_name: str = "llm_chat",
            no_remote_api: bool = None,
    ):
        '''
        对应api.py/chat/chat接口
        '''
        if no_remote_api is None:
            no_remote_api = self.no_remote_api

        data = {
            "query": query,
            "history": history,
            "stream": stream,
            "model_name": model,
            "temperature": temperature,
            "prompt_name": prompt_name,
        }

        print(f"received input message:")
        pprint(data)

        if no_remote_api:
            from server.chat.chat import chat
            response = run_async(chat(**data))
            return self._fastapi_stream2generator(response)
        else:
            response = self.post("/chat/chat", json=data, stream=True)
            return self._httpx_stream2generator(response)

    def agent_chat(
            self,
            query: str,
            history: List[Dict] = [],
            stream: bool = True,
            model: str = LLM_MODEL,
            temperature: float = TEMPERATURE,
            no_remote_api: bool = None,
    ):
        '''
        对应api.py/chat/agent_chat 接口
        '''
        if no_remote_api is None:
            no_remote_api = self.no_remote_api

        data = {
            "query": query,
            "history": history,
            "stream": stream,
            "model_name": model,
            "temperature": temperature,
        }

        print(f"received input message:")
        pprint(data)

        if no_remote_api:
            from server.chat.agent_chat import agent_chat
            response = run_async(agent_chat(**data))
            return self._fastapi_stream2generator(response)
        else:
            response = self.post("/chat/agent_chat", json=data, stream=True)
            return self._httpx_stream2generator(response)

    def knowledge_base_chat(
            self,
            query: str,
            knowledge_base_name: str,
            top_k: int = VECTOR_SEARCH_TOP_K,
            score_threshold: float = SCORE_THRESHOLD,
            history: List[Dict] = [],
            stream: bool = True,
            model: str = LLM_MODEL,
            temperature: float = TEMPERATURE,
            prompt_name: str = "knowledge_base_chat",
            no_remote_api: bool = None,
    ):
        '''
        对应api.py/chat/knowledge_base_chat接口
        '''
        if no_remote_api is None:
            no_remote_api = self.no_remote_api

        data = {
            "query": query,
            "knowledge_base_name": knowledge_base_name,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "history": history,
            "stream": stream,
            "model_name": model,
            "temperature": temperature,
            "local_doc_url": no_remote_api,
            "prompt_name": prompt_name,
        }

        print(f"received input message:")
        pprint(data)

        if no_remote_api:
            from server.chat.knowledge_base_chat import knowledge_base_chat
            response = run_async(knowledge_base_chat(**data))
            return self._fastapi_stream2generator(response, as_json=True)
        else:
            response = self.post(
                "/chat/knowledge_base_chat",
                json=data,
                stream=True,
            )
            return self._httpx_stream2generator(response, as_json=True)

    def summary_chat(
            self,
            knowledge_base_name: str,
            file: List[Union[str, Path, bytes]] = None,
            stream: bool = True,
            model: str = LLM_MODEL,
            temperature: float = TEMPERATURE,
            no_remote_api: bool = None,
    ):
        '''
        对应api.py/chat/summary_chat接口
        '''

        logger.info(" ================ 进 summary_chat 接口了 ======================")
        logger.info("knowledge_base_name: " + knowledge_base_name)
        logger.info(file)

        if no_remote_api is None:
            no_remote_api = self.no_remote_api

        data = {
            "knowledge_base_name": knowledge_base_name,
            "stream": stream,
            "model_name": model,
            "temperature": temperature,
            "local_doc_url": no_remote_api,
        }

        print(f"received input message:")
        pprint(data)

        if no_remote_api:
            from server.chat.summary_chat import summary_chat
            response = run_async(summary_chat(**data))
            return self._fastapi_stream2generator(response, as_json=True)
        if file is None:
            '''
            给 json参数 传参，content-type 为 application/json
            '''
            response = self.post(
                "/chat/summary_chat",
                data=data,
                stream=True,
            )
            return self._httpx_stream2generator(response, as_json=True)
        else:
            '''
            给 data参数 传参，content-type 为 application/x-www-form-urlencoded
            '''
            file = [convert_file(f) for f in file]

            logger.info(" ================ 转换完file了 ======================")
            logger.info("knowledge_base_name: " + knowledge_base_name)
            logger.info(file)

            response = self.post(
                "/chat/summary_chat",
                data=data,
                stream=True,
                files=[("files", (filename, file_content)) for filename, file_content in file],
            )
            return self._httpx_stream2generator(response, as_json=True)

    def search_engine_chat(
            self,
            query: str,
            search_engine_name: str,
            top_k: int = SEARCH_ENGINE_TOP_K,
            history: List[Dict] = [],
            stream: bool = True,
            model: str = LLM_MODEL,
            temperature: float = TEMPERATURE,
            prompt_name: str = "knowledge_base_chat",
            no_remote_api: bool = None,
    ):
        '''
        对应api.py/chat/search_engine_chat接口
        '''
        if no_remote_api is None:
            no_remote_api = self.no_remote_api

        data = {
            "query": query,
            "search_engine_name": search_engine_name,
            "top_k": top_k,
            "history": history,
            "stream": stream,
            "model_name": model,
            "temperature": temperature,
            "prompt_name": prompt_name,
        }

        print(f"received input message:")
        pprint(data)

        if no_remote_api:
            from server.chat.search_engine_chat import search_engine_chat
            response = run_async(search_engine_chat(**data))
            return self._fastapi_stream2generator(response, as_json=True)
        else:
            response = self.post(
                "/chat/search_engine_chat",
                json=data,
                stream=True,
            )
            return self._httpx_stream2generator(response, as_json=True)

    # 知识库相关操作

    def _check_httpx_json_response(
            self,
            response: httpx.Response,
            errorMsg: str = f"无法连接API服务器，请确认已执行python server\\api.py",
    ) -> Dict:
        '''
        check whether httpx returns correct data with normal Response.
        error in api with streaming support was checked in _httpx_stream2enerator
        '''
        try:
            return response.json()
        except Exception as e:
            msg = "API未能返回正确的JSON。" + (errorMsg or str(e))
            logger.error(f'{e.__class__.__name__}: {msg}',
                         exc_info=e if log_verbose else None)
            return {"code": 500, "msg": msg}

    def list_video_path(
            self,
            knowledge_base_name: str,
            no_remote_api: bool = None,
    ):
        logger.info(" ================ 进 list_video_path 接口了 ======================")
        logger.info("knowledge_base_name: " + knowledge_base_name)
        '''
        对应api.py/knowledge_base/list_video_path接口
        '''
        if no_remote_api is None:
            no_remote_api = self.no_remote_api

        data = {
            "knowledge_base_name": knowledge_base_name,
        }

        if no_remote_api:
            from server.knowledge_base.kb_api import list_video_path
            response = list_video_path(knowledge_base_name)
            return response.data
        else:
            response = self.get(
                "/knowledge_base/list_video_path",
                params=data
            )
            data = self._check_httpx_json_response(response)
            return data.get("data", [])

    def list_knowledge_bases(
            self,
            no_remote_api: bool = None,
    ):
        '''
        对应api.py/knowledge_base/list_knowledge_bases接口
        '''
        if no_remote_api is None:
            no_remote_api = self.no_remote_api

        if no_remote_api:
            from server.knowledge_base.kb_api import list_kbs
            response = list_kbs()
            return response.data
        else:
            response = self.get("/knowledge_base/list_knowledge_bases")
            data = self._check_httpx_json_response(response)
            return data.get("data", [])

    def create_knowledge_base(
            self,
            knowledge_base_name: str,
            vector_store_type: str = "faiss",
            embed_model: str = EMBEDDING_MODEL,
            no_remote_api: bool = None,
    ):
        '''
        对应api.py/knowledge_base/create_knowledge_base接口
        '''
        if no_remote_api is None:
            no_remote_api = self.no_remote_api

        data = {
            "knowledge_base_name": knowledge_base_name,
            "vector_store_type": vector_store_type,
            "embed_model": embed_model,
        }

        if no_remote_api:
            from server.knowledge_base.kb_api import create_kb
            response = create_kb(**data)
            return response.dict()
        else:
            response = self.post(
                "/knowledge_base/create_knowledge_base",
                json=data,
            )
            return self._check_httpx_json_response(response)

    def delete_knowledge_base(
            self,
            knowledge_base_name: str,
            no_remote_api: bool = None,
    ):
        '''
        对应api.py/knowledge_base/delete_knowledge_base接口
        '''
        if no_remote_api is None:
            no_remote_api = self.no_remote_api

        if no_remote_api:
            from server.knowledge_base.kb_api import delete_kb
            response = delete_kb(knowledge_base_name)
            return response.dict()
        else:
            response = self.post(
                "/knowledge_base/delete_knowledge_base",
                json=f"{knowledge_base_name}",
            )
            return self._check_httpx_json_response(response)

    def list_kb_docs(
            self,
            knowledge_base_name: str,
            no_remote_api: bool = None,
    ):
        '''
        对应api.py/knowledge_base/list_files接口
        '''
        if no_remote_api is None:
            no_remote_api = self.no_remote_api

        if no_remote_api:
            from server.knowledge_base.kb_doc_api import list_files
            response = list_files(knowledge_base_name)
            return response.data
        else:
            response = self.get(
                "/knowledge_base/list_files",
                params={"knowledge_base_name": knowledge_base_name}
            )
            data = self._check_httpx_json_response(response)
            return data.get("data", [])

    def search_kb_docs(
            self,
            query: str,
            knowledge_base_name: str,
            top_k: int = VECTOR_SEARCH_TOP_K,
            score_threshold: int = SCORE_THRESHOLD,
            no_remote_api: bool = None,
    ) -> List:
        '''
        对应api.py/knowledge_base/search_docs接口
        '''
        if no_remote_api is None:
            no_remote_api = self.no_remote_api

        data = {
            "query": query,
            "knowledge_base_name": knowledge_base_name,
            "top_k": top_k,
            "score_threshold": score_threshold,
        }

        if no_remote_api:
            from server.knowledge_base.kb_doc_api import search_docs
            return search_docs(**data)
        else:
            response = self.post(
                "/knowledge_base/search_docs",
                json=data,
            )
            data = self._check_httpx_json_response(response)
            return data

    # @async_decorator
    async def upload_kb_docs(
            self,
            knowledge_base_name: str,
            files: List[Union[str, Path, bytes]] = None,
            link: Tuple[str, str] = None,
            override: bool = False,
            to_vector_store: bool = True,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=OVERLAP_SIZE,
            zh_title_enhance=ZH_TITLE_ENHANCE,
            docs: Dict = {},
            not_refresh_vs_cache: bool = False,
            no_remote_api: bool = None,
    ):
        logger.info("=================== upload start =================")
        logger.info(link)

        '''
        对应api.py/knowledge_base/upload_docs接口
        '''
        if no_remote_api is None:
            no_remote_api = self.no_remote_api
        if files is not None:
            files = [convert_file(file) for file in files]
        data = {
            "knowledge_base_name": knowledge_base_name,
            "override": override,
            "to_vector_store": to_vector_store,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "zh_title_enhance": zh_title_enhance,
            "docs": docs,
            "not_refresh_vs_cache": not_refresh_vs_cache,
            "link": json.dumps(link),
        }

        if no_remote_api:
            from server.knowledge_base.kb_doc_api import upload_docs
            from fastapi import UploadFile
            from tempfile import SpooledTemporaryFile

            upload_files = []
            for filename, file in files:
                temp_file = SpooledTemporaryFile(max_size=10 * 1024 * 1024)
                temp_file.write(file.read())
                temp_file.seek(0)
                upload_files.append(UploadFile(file=temp_file, filename=filename))

            response = upload_docs(upload_files, **data)
            return response.dict()
        else:
            if isinstance(data["docs"], dict):
                data["docs"] = json.dumps(data["docs"], ensure_ascii=False)

            # 更新 docs
            # response = await self.apost(
            #     "/knowledge_base/upload_docs",
            #     data=data,
            #     files=[("files", (filename, file)) for filename, file in files],
            # )
            # logger.info("=================== upload end =================")
            # return self._check_httpx_json_response(response)

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url=self._parse_url("/knowledge_base/upload_docs"),
                    data=data,
                    files=[("files", (filename, file)) for filename, file in files],
                )
                logger.info("=================== upload end =================")
                return response
            # return self._check_httpx_json_response(response)

    def delete_kb_docs(
            self,
            knowledge_base_name: str,
            file_names: List[str],
            delete_content: bool = False,
            not_refresh_vs_cache: bool = False,
            no_remote_api: bool = None,
    ):
        '''
        对应api.py/knowledge_base/delete_docs接口
        '''
        if no_remote_api is None:
            no_remote_api = self.no_remote_api

        data = {
            "knowledge_base_name": knowledge_base_name,
            "file_names": file_names,
            "delete_content": delete_content,
            "not_refresh_vs_cache": not_refresh_vs_cache,
        }

        if no_remote_api:
            from server.knowledge_base.kb_doc_api import delete_docs
            response = delete_docs(**data)
            return response.dict()
        else:
            response = self.post(
                "/knowledge_base/delete_docs",
                json=data,
            )
            return self._check_httpx_json_response(response)

    def update_kb_docs(
            self,
            knowledge_base_name: str,
            file_names: List[str],
            override_custom_docs: bool = False,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=OVERLAP_SIZE,
            zh_title_enhance=ZH_TITLE_ENHANCE,
            docs: Dict = {},
            not_refresh_vs_cache: bool = False,
            no_remote_api: bool = None,
    ):
        '''
        对应api.py/knowledge_base/update_docs接口
        '''
        if no_remote_api is None:
            no_remote_api = self.no_remote_api

        data = {
            "knowledge_base_name": knowledge_base_name,
            "file_names": file_names,
            "override_custom_docs": override_custom_docs,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "zh_title_enhance": zh_title_enhance,
            "docs": docs,
            "not_refresh_vs_cache": not_refresh_vs_cache,
        }
        if no_remote_api:
            from server.knowledge_base.kb_doc_api import update_docs
            response = update_docs(**data)
            return response.dict()
        else:
            if isinstance(data["docs"], dict):
                data["docs"] = json.dumps(data["docs"], ensure_ascii=False)
            response = self.post(
                "/knowledge_base/update_docs",
                json=data,
            )
            return self._check_httpx_json_response(response)

    def recreate_vector_store(
            self,
            knowledge_base_name: str,
            allow_empty_kb: bool = True,
            vs_type: str = DEFAULT_VS_TYPE,
            embed_model: str = EMBEDDING_MODEL,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=OVERLAP_SIZE,
            zh_title_enhance=ZH_TITLE_ENHANCE,
            no_remote_api: bool = None,
    ):
        '''
        对应api.py/knowledge_base/recreate_vector_store接口
        '''
        if no_remote_api is None:
            no_remote_api = self.no_remote_api

        data = {
            "knowledge_base_name": knowledge_base_name,
            "allow_empty_kb": allow_empty_kb,
            "vs_type": vs_type,
            "embed_model": embed_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "zh_title_enhance": zh_title_enhance,
        }

        if no_remote_api:
            from server.knowledge_base.kb_doc_api import recreate_vector_store
            response = recreate_vector_store(**data)
            return self._fastapi_stream2generator(response, as_json=True)
        else:
            response = self.post(
                "/knowledge_base/recreate_vector_store",
                json=data,
                stream=True,
                timeout=None,
            )
            return self._httpx_stream2generator(response, as_json=True)

    # LLM模型相关操作
    def list_running_models(
            self,
            controller_address: str = None,
            no_remote_api: bool = None,
    ):
        '''
        获取Fastchat中正运行的模型列表
        '''
        if no_remote_api is None:
            no_remote_api = self.no_remote_api

        data = {
            "controller_address": controller_address,
        }
        if no_remote_api:
            from server.llm_api import list_running_models
            return list_running_models(**data).data
        else:
            r = self.post(
                "/llm_model/list_running_models",
                json=data,
            )
            return r.json().get("data", [])

    def list_config_models(self, no_remote_api: bool = None) -> Dict[str, List[str]]:
        '''
        获取configs中配置的模型列表，返回形式为{"type": [model_name1, model_name2, ...], ...}。
        如果no_remote_api=True, 从运行ApiRequest的机器上获取；否则从运行api.py的机器上获取。
        '''
        if no_remote_api is None:
            no_remote_api = self.no_remote_api

        if no_remote_api:
            from server.llm_api import list_config_models
            return list_config_models().data
        else:
            r = self.post(
                "/llm_model/list_config_models",
            )
            return r.json().get("data", {})

    def stop_llm_model(
            self,
            model_name: str,
            controller_address: str = None,
            no_remote_api: bool = None,
    ):
        '''
        停止某个LLM模型。
        注意：由于Fastchat的实现方式，实际上是把LLM模型所在的model_worker停掉。
        '''
        if no_remote_api is None:
            no_remote_api = self.no_remote_api

        data = {
            "model_name": model_name,
            "controller_address": controller_address,
        }

        if no_remote_api:
            from server.llm_api import stop_llm_model
            return stop_llm_model(**data).dict()
        else:
            r = self.post(
                "/llm_model/stop",
                json=data,
            )
            return r.json()

    def change_llm_model(
            self,
            model_name: str,
            new_model_name: str,
            controller_address: str = None,
            no_remote_api: bool = None,
    ):
        '''
        向fastchat controller请求切换LLM模型。
        '''
        if no_remote_api is None:
            no_remote_api = self.no_remote_api

        if not model_name or not new_model_name:
            return

        running_models = self.list_running_models()
        if new_model_name == model_name or new_model_name in running_models:
            return {
                "code": 200,
                "msg": "无需切换"
            }

        if model_name not in running_models:
            return {
                "code": 500,
                "msg": f"指定的模型'{model_name}'没有运行。当前运行模型：{running_models}"
            }

        config_models = self.list_config_models()
        if new_model_name not in config_models.get("local", []):
            return {
                "code": 500,
                "msg": f"要切换的模型'{new_model_name}'在configs中没有配置。"
            }

        data = {
            "model_name": model_name,
            "new_model_name": new_model_name,
            "controller_address": controller_address,
        }

        if no_remote_api:
            from server.llm_api import change_llm_model
            return change_llm_model(**data).dict()
        else:
            r = self.post(
                "/llm_model/change",
                json=data,
                timeout=HTTPX_DEFAULT_TIMEOUT,  # wait for new worker_model
            )
            return r.json()


def check_error_msg(data: Union[str, dict, list], key: str = "errorMsg") -> str:
    '''
    return error message if error occured when requests API
    '''
    if isinstance(data, dict):
        if key in data:
            return data[key]
        if "code" in data and data["code"] != 200:
            return data["msg"]
    return ""


def check_success_msg(data: Union[str, dict, list], key: str = "msg") -> str:
    '''
    return error message if error occured when requests API
    '''
    if (isinstance(data, dict)
            and key in data
            and "code" in data
            and data["code"] == 200):
        return data[key]
    return ""


if __name__ == "__main__":
    api = ApiRequest(no_remote_api=True)

    api.knowledge_base_chat("what is ai?",
                            knowledge_base_name="samples",
                            model="gpt-3.5-turbo",
                            no_remote_api=False)

    # print(api.chat_fastchat(
    #     messages=[{"role": "user", "content": "hello"}]
    # ))

    # with api.chat_chat("你好") as r:
    #     for t in r.iter_text(None):
    #         print(t)

    # r = api.chat_chat("你好", no_remote_api=True)
    # for t in r:
    #     print(t)

    # r = api.duckduckgo_search_chat("室温超导最新研究进展", no_remote_api=True)
    # for t in r:
    #     print(t)

    # print(api.list_knowledge_bases())
