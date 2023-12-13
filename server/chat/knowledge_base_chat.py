import aiofiles
from fastapi import Body, Request
from fastapi.responses import StreamingResponse
from configs import (LLM_MODEL, VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD, TEMPERATURE, logger)
from server.utils import wrap_done, get_ChatOpenAI
from server.utils import BaseResponse, get_prompt_template
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable, List, Optional
import asyncio
from langchain.prompts.chat import ChatPromptTemplate
from server.chat.utils import History
from server.knowledge_base.kb_service.base import KBService, KBServiceFactory
import json
import os
from urllib.parse import urlencode
from server.knowledge_base.kb_doc_api import search_docs

recoed_question_locks = {}  # Global locks for recorded questions
recoed_question_locks_lock = None  # Lock for the global dictionary of locks


async def knowledge_base_chat(query: str = Body(..., description="用户输入", examples=["你好"]),
                              knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
                              top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                              score_threshold: float = Body(SCORE_THRESHOLD,
                                                            description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右",
                                                            ge=0, le=1),
                              history: List[History] = Body([],
                                                            description="历史对话",
                                                            examples=[[
                                                                {"role": "user",
                                                                 "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                                                {"role": "assistant",
                                                                 "content": "虎头虎脑"}]]
                                                            ),
                              stream: bool = Body(False, description="流式输出"),
                              model_name: str = Body(LLM_MODEL, description="LLM 模型名称。"),
                              temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                              prompt_name: str = Body("knowledge_base_chat",
                                                      description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
                              local_doc_url: bool = Body(False, description="知识文件返回本地路径(true)或URL(false)"),
                              request: Request = None,
                              ):
    """
        This function is an asynchronous function that handles the chat with the knowledge base.
        It takes in several parameters including the query, knowledge base name, top_k, score_threshold, history, stream, model_name, temperature, prompt_name, local_doc_url, and request.
        It returns a StreamingResponse which is a response that can be streamed to the client.
    """
    global recoed_question_locks_lock
    if recoed_question_locks_lock is None:
        recoed_question_locks_lock = asyncio.Lock()
    # 验证文件
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    history = [History.from_data(h) for h in history]

    async def get_lock(file_path):
        """
            This function is an asynchronous function that gets the lock for a given file path.
            It takes in a file path as a parameter and returns the lock for that file path.
            If the lock does not exist, it creates a new one.
        """
        # 设计：
        # 减少全局锁的依赖：
        # 在原始方案中，无论协程是访问同一个文件还是不同的文件，都必须通过全局锁来获取或创建相应的锁。
        # 这意味着即使两个协程操作不同的文件，它们在获取锁时也必须相互等待，从而限制了并发性能。
        # 在优化后的方案中，我们尽量减少了对全局锁的依赖。
        # 只有在首次创建某个文件的锁信息时才需要使用全局锁，一旦创建完成，就使用该文件路径的专用小锁来控制对大锁的访问。
        # 这大大减少了锁的竞争，提高了并发性能。

        # 需要考虑极小的竞态条件风险（例如，锁很少被创建或竞争不激烈）：
        # 两个并发执行的 get_lock 调用可能同时发现字典中没有锁，然后都尝试添加锁。
        # 由于存在多个锁，不同协程可能会同时获得不同的锁，进而同时尝试对同一个文件进行写入操作。
        # 为了避免这种情况，我们使用一个全局锁来确保一次只有一个线程可以 检查 和 创建锁。
        async with recoed_question_locks_lock:
            if file_path not in recoed_question_locks:
                recoed_question_locks[file_path] = (asyncio.Lock(), asyncio.Lock())  # (文件锁, 小锁)

        _, path_lock = recoed_question_locks[file_path]
        async with path_lock:  # 使用文件路径的专用小锁
            if recoed_question_locks[file_path][0] is None:
                recoed_question_locks[file_path] = (asyncio.Lock(), path_lock)  # 初始化文件锁

        return recoed_question_locks[file_path][0]

    async def record_question(prompt: str, selected_kb: str):
        logger.info(f"======= record start =========")
        # Validate the inputs
        if not isinstance(prompt, str) or not isinstance(selected_kb, str):
            raise ValueError('Both prompt and selected_kb must be strings')

        # Construct the file path
        file_path = os.path.join('knowledge_base', selected_kb, 'qa.txt')

        # Get the lock for this file path, or create a new one if it doesn't exist
        lock = await get_lock(file_path)

        # Acquire the lock
        async with lock:
            try:
                # Open the file in append mode
                async with aiofiles.open(file_path, mode='a') as f:
                    # Write the prompt to the file
                    await f.write(prompt + '\n')
                    logger.info(f"======= Successfully wrote to file {file_path} =======")
            except Exception as e:
                # Log the error
                logger.info(f"======= Error writing to file {file_path}: {e} =======")
                raise
            finally:
                # Ensure the file is closed
                if not f.closed:
                    await f.close()
                    logger.info(f"======= Successfully closed file {file_path} =======")

    def get_chain(history: Optional[List[History]],
                  model_name: str = LLM_MODEL,
                  prompt_name: str = prompt_name,
                  stream: bool = True,
                  ) -> (LLMChain, AsyncIteratorCallbackHandler):
        callback = None
        if stream:
            callback = AsyncIteratorCallbackHandler()
            model = get_ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                callbacks=[callback],
            )
        else:
            model = get_ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                stream=False,
            )
        # （history）
        # 拿到prompt模版
        prompt_template = get_prompt_template(prompt_name)
        # 把prompt模版转换成ChatMessagePromptTemplate（langchain里的对象）
        input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        # 1.把history里的所有内容模版转换成ChatMessagePromptTemplate，加入消息列表
        # 2.把这次的模板input_msg加入消息列表
        # 3.把消息列表(1和2)一块转换成 ChatPromptTemplate（一个包含所有消息的字符串）
        # （ChatPromptTemplate 是 langchain 库中的一个类，用于生成聊天模板。它接受一个消息列表作为输入，返回一个包含所有消息的字符串。）
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])
        # 创建 chain
        chain = LLMChain(prompt=chat_prompt, llm=model)
        return chain, callback

    async def knowledge_base_chat_iterator(query: str,
                                           top_k: int,
                                           history: Optional[List[History]],
                                           model_name: str = LLM_MODEL,
                                           prompt_name: str = prompt_name,
                                           ) -> AsyncIterable[str]:
        '''
        任务列表：
            - 任务1：问gpt 用户的 detail_question（问题到底在问什么内容）
            - 任务2：根据 detail_question 让gpt提取用户问题
            - 任务3：根据 detail_question + “Please provide an answer.” 检索出所有的文本，放到 context 里
                - 用 context 和 组合问题 问gpt
            - 任务4：保存gpt提取的用户问题

        要求：
            - 任务1 单独执行
            - 任务2 和 任务3同步（必须在 任务1 之后 执行）
            - 任务4 必须在 任务2 之后，不需要等待 任务3
        '''

        chain1, _ = get_chain(history, model_name, "get_detail_question", stream=False)
        task1 = asyncio.create_task(chain1.acall({"question": query}))
        detail_question = ""
        try:
            detail_question_request = await task1
        except Exception as e:
            detail_question = query
            logger.error(f"task1 failed with error: {e}")
        else:
            # Continue with the rest of the code that depends on task1's result
            if detail_question_request is None or detail_question_request == {}:
                detail_question = query
            else:
                detail_question = detail_question_request['text']

        # 组合问题
        detail_question = f"{detail_question} Please provide an answer."

        logger.info(f"======= detail_question: {detail_question} =========")

        task_list = []  # 任务列表

        chain2, _ = get_chain(history, model_name, "question_extraction", stream=False)
        task2 = asyncio.create_task(chain2.acall({"question": detail_question}))
        # task_list.append(task2)
        extract_question = ""
        try:
            extract_question_request = await task2
        except Exception as e:
            extract_question = query
            logger.error(f"task2 failed with error: {e}")
        else:
            # Continue with the rest of the code that depends on task1's result
            if extract_question_request is None or extract_question_request == {}:
                extract_question = query
            else:
                extract_question = extract_question_request['text']
                if extract_question.startswith("user's request: "):
                    extract_question = extract_question[16:]
        logger.info(f"======= extract_question: {extract_question} =========")

        # (reference)检索出所有的文本，放到 context 里
        docs = search_docs(detail_question, knowledge_base_name, top_k, score_threshold)
        context = "\n".join([doc.page_content for doc in docs])

        chain3, callback3 = get_chain(history, model_name, "knowledge_base_chat")

        # Begin a task that runs in the background.
        # asyncio.create_task()：用于创建一个Task对象。
        #   Task对象代表一个协程（coroutine）的执行，可以异步执行协程函数，并在需要时暂停和恢复执行。
        #   通过create_task()方法创建的Task对象可以被添加到事件循环中，由事件循环调度执行。
        # wrap_done()：将一个可等待对象与一个事件包装起来，以在完成或引发异常时发出信号。
        # acall()：异步执行链。
        #   形参：
        #       inputs - 输入的字典，或者如果链只期望一个参数，则为单个输入。应包含在Chain.input_keys中指定的所有输入，但不包括将由链的内存设置的输入。
        #       return_only_outputs - 是否仅在响应中返回输出。如果为True，则仅返回此链生成的新键。如果为False，则返回输入键和此链生成的新键。默认值为False。
        #       callbacks - 用于此链运行的回调函数。除了在构建过程中传递给链的回调函数之外，还会调用这些运行时回调函数，但只有这些运行时回调标签才会传播到对其他对象的调用中。
        #       tags - 要传递给所有回调函数的字符串标签列表。除了在构建过程中传递给链的标签之外，还将传递这些运行时标签，但只有这些运行时标签才会传播到对其他对象的调用中。
        #       metadata - 与该链接关联的可选元数据，默认值为None
        #       include_run_info - 是否在响应中包含运行信息，默认值为False。
        #   返回值：
        #       命名输出字典。应包含Chain.output_keys指定 的所有输出
        # AsyncIteratorCallbackHandler().done 是 AsyncIteratorCallbackHandler 类的一个属性它是一个 asyncio.Event 对象。
        #   在 wrap_done 里：当 AsyncIteratorCallbackHandler 对象完成迭代时，它会设置为 True。
        #   这个属性可以用于检查异步迭代器是否完成
        task3 = asyncio.create_task(wrap_done(
            chain3.acall({"context": context, "question": detail_question}),
            callback3.done),
        )
        # task_list.append(task3)

        # 同时生成来源文档
        source_documents = []
        for inum, doc in enumerate(docs):
            filename = os.path.split(doc.metadata["source"])[-1]
            if local_doc_url:
                url = "file://" + doc.metadata["source"]
            else:
                parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
                url = f"{request.base_url}knowledge_base/download_doc?" + parameters
            text = f"""from [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
            source_documents.append(text)

        # Optionally wait for task3 if you need to ensure it completes
        # 一个 token, 一个 token 的返回：
        if stream:
            # 异步迭代一个异步迭代器 callback.aiter()，并将每个元素赋值给 token
            async for token in callback3.aiter():
                # Use server-sent-events to stream the response
                # 使用 json.dumps() 将 token 转换为 JSON 格式；{'answer': '能'}
                # 使用 yield 语句将 JSON 格式的令牌作为响应流式传输给客户端
                yield json.dumps({"answer": token}, ensure_ascii=False)
            # 使用 yield 语句将 JSON 格式的 source_documents 作为响应流式传输给客户端
            yield json.dumps({"docs": source_documents}, ensure_ascii=False)
        # 拿到所有的 answer 再和 source 一块返回：
        else:
            answer = ""
            async for token in callback3.aiter():
                answer += token
            yield json.dumps({"answer": answer,
                              "docs": source_documents},
                             ensure_ascii=False)

        # await asyncio.gather(*task_list)

        # Wait for task3 to finish before starting task4
        # 记录问题
        task4 = asyncio.create_task(record_question(extract_question, knowledge_base_name))
        await task4

        await task3

    return StreamingResponse(knowledge_base_chat_iterator(query=query,
                                                          top_k=top_k,
                                                          history=history,
                                                          model_name=model_name,
                                                          prompt_name=prompt_name),
                             media_type="text/event-stream")
