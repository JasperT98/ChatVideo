import tiktoken
from io import BytesIO
from pathlib import Path
from typing import *
from fastapi import Body, Request, File, Form, Body, Query, UploadFile
from fastapi.responses import StreamingResponse
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import TokenTextSplitter
from langchain.schema import Document
import numpy as np
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor, as_completed

from configs import (LLM_MODEL, VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD, TEMPERATURE, logger, log_verbose,)
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


def token_counter(text: str):
    """
    Count the number of tokens in a string of text.

    :param text: The text to count the tokens of.

    :return: The number of tokens in the text.
    """
    encoding = tiktoken.get_encoding('cl100k_base')
    token_list = encoding.encode(text, disallowed_special=())
    tokens = len(token_list)
    return tokens


def doc_to_text(document):
    """
    Convert a langchain Document object into a string of text.

    :param document: The loaded langchain Document object to convert.

    :return: A string of text.
    """
    # logger.info("================ 进 doc_to_text 函数了 ======================")
    # logger.info(document)
    # logger.info(document.page_content)
    text = ''
    for i in document:
        text += i.page_content
    special_tokens = ['>|endoftext|', '<|fim_prefix|', '<|fim_middle|', '<|fim_suffix|', '<|endofprompt|']
    words = text.split()
    filtered_words = [word for word in words if word not in special_tokens]
    text = ' '.join(filtered_words)
    return text


def token_limit(doc, maximum=200000):
    """
    Check if a document has more tokens than a specified maximum.

    :param doc: The langchain Document object to check.

    :param maximum: The maximum number of tokens allowed.

    :return: True if the document has less than the maximum number of tokens, False otherwise.
    """
    text = doc_to_text(doc)
    count = token_counter(text)
    print(count)
    if count > maximum:
        return False
    return True


def token_minimum(doc, minimum=2000):
    """
    Check if a document has more tokens than a specified minimum.

    :param doc: The langchain Document object to check.

    :param minimum: The minimum number of tokens allowed.

    :return: True if the document has more than the minimum number of tokens, False otherwise.
    """
    text = doc_to_text(doc)
    count = token_counter(text)
    if count < minimum:
        return False
    return True


def validate_doc_size(doc):
    """
    Validates the size of the document

    :param doc: doc to validate

    :return: True if the doc is valid, False otherwise
    """
    if not token_limit(doc, 800000):
        logger.warning('File or transcript too big!')
        return False

    if not token_minimum(doc, 500):
        logger.warning('File or transcript too small!')
        return False
    return True


async def summary_chat(knowledge_base_name: str = Form(None, description="知识库名称", examples=["samples"]),
                       files: List[UploadFile] = File(None, description="上传的字幕文件，默认是单个文件"),
                       stream: bool = Form(False, description="流式输出"),
                       model_name: str = Form(LLM_MODEL, description="LLM 模型名称。"),
                       temperature: float = Form(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                       local_doc_url: bool = Form(False, description="知识文件返回本地路径(true)或URL(false)"),
                       request: Request = None,
                       find_clusters: bool = Form(False,
                                                  description="Whether to find optimal clusters or not, experimental")
                       ):
    logger.info(" ================ 进 summery 函数了 ======================")
    logger.info("knowledge_base_name: " + knowledge_base_name)
    logger.info(files)
    if knowledge_base_name is None:
        return BaseResponse(code=404, msg=f"未指定知识库和文件")
    '''
    配置各种变量：
    '''
    # 验证文件
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")
    # 获取这个 kb 的 embedding模型（从 cache 里获得）
    embedding = kb._load_embeddings()
    doc = None
    if files is None or files == [] or files == [None]:
        '''
        如果没有给文件就找知识库里的transcript文件，并且加载出响应的 doc
        '''
        logger.info(" ================ 没有给文件, 找知识库里的transcript文件，并且加载出相应的 doc ======================")
        all_files_in_given_kb = kb.list_files()
        name = None
        for f in all_files_in_given_kb:
            if "transcript" in f:
                name = f
            else:
                return BaseResponse(code=404, msg=f"{knowledge_base_name}中未找到字幕文件")
        # 加载 doc 是 list[Document]
        doc = kb.list_docs(name)
    else:
        '''
        如果给了文件就用给的文件生成summery，并且生成对应的 doc
        '''
        logger.info(
            " ================ 给了文件, 用给的文件生成summery ======================")
        for f in files:
            # 读出 bytes，并将 bytes 转换成 str
            file_content = f.file.read().decode("utf-8")
            doc = [Document(page_content=file_content, metadata={"source": "upload"})]

    map_prompt = get_prompt_template("file_map")
    combine_prompt = get_prompt_template("file_combine")

    logger.info(" ================ 开始验证doc ======================")
    if doc is None:
        return BaseResponse(code=404, msg=f"doc为空")
    if not validate_doc_size(doc):
        # doc的大小不符合要求，返回
        return BaseResponse(code=404, msg=f"doc大小不符合要求")

    def map_vectors_to_docs(indices, docs):
        """
        Map a list of indices to a list of loaded langchain Document objects.

        :param indices: A list of indices to map.

        :param docs: A list of langchain Document objects to map to.

        :return: A list of loaded langchain Document objects.
        """
        selected_docs = [docs[i] for i in indices]
        return selected_docs

    def get_closest_vectors(vectors, kmeans):
        """
        Get the closest vectors to the cluster centers of a K-Means clustering object.

        :param vectors: A list of vectors to cluster.

        :param kmeans: A K-Means clustering object.

        :return: A list of indices of the closest vectors to the cluster centers.
        """
        closest_indices = []
        for i in range(len(kmeans.cluster_centers_)):
            distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
            closest_index = np.argmin(distances)
            closest_indices.append(closest_index)

        selected_indices = sorted(closest_indices)
        return selected_indices

    def determine_optimal_clusters(inertia_values):
        """
        Determine the optimal number of clusters to use based on the inertia values.

        :param inertia_values: A list of inertia values.

        :return: The optimal number of clusters to use.
        """
        distances = []
        for i in range(len(inertia_values) - 1):
            p1 = np.array([i + 1, inertia_values[i]])
            p2 = np.array([i + 2, inertia_values[i + 1]])
            d = np.linalg.norm(np.cross(p2 - p1, p1 - np.array([1, 0]))) / np.linalg.norm(p2 - p1)
            distances.append(d)
        optimal_clusters = distances.index(max(distances)) + 2
        return optimal_clusters

    def calculate_inertia(vectors, max_clusters=12):
        """
        Calculate the inertia values for a range of clusters.

        :param vectors: A list of vectors to cluster.

        :param max_clusters: The maximum number of clusters to use.

        :return: A list of inertia values.
        """
        inertia_values = []
        for num_clusters in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)
            inertia_values.append(kmeans.inertia_)
        return inertia_values

    def kmeans_clustering(vectors, num_clusters=None):
        """
        Cluster a list of vectors using K-Means clustering.

        :param vectors: A list of vectors to cluster.

        :param num_clusters: The number of clusters to use. If None, the optimal number of clusters will be determined.

        :return: A K-Means clustering object.
        """
        if num_clusters is None:
            inertia_values = calculate_inertia(vectors)
            num_clusters = determine_optimal_clusters(inertia_values)
            print(f'Optimal number of clusters: {num_clusters}')

        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)
        return kmeans

    def remove_special_tokens(docs):
        special_tokens = ['>|endoftext|', '<|fim_prefix|', '<|fim_middle|', '<|fim_suffix|', '<|endofprompt|>']
        for doc in docs:
            content = doc.page_content
            for special in special_tokens:
                content = content.replace(special, '')
                doc.page_content = content
        return docs

    def embed_docs(docs):
        """
        Embed a list of documents into a list of vectors.

        :param docs: A list of documents to embed.

        :param api_key: The OpenAI API key to use for embedding.

        :return: A list of vectors.
        """
        docs = remove_special_tokens(docs)
        vectors = embedding.embed_documents([x.page_content for x in docs])

        return vectors

    def split_by_tokens(doc, num_clusters, ratio=5, minimum_tokens=200, maximum_tokens=2000):
        """
        Split a  langchain Document object into a list of smaller langchain Document objects.

        :param doc: The langchain Document object to split.

        :param num_clusters: The number of clusters to use.

        :param ratio: The ratio of documents to clusters to use for splitting.

        :param minimum_tokens: The minimum number of tokens to use for splitting.

        :param maximum_tokens: The maximum number of tokens to use for splitting.

        :return: A list of langchain Document objects.
        """
        text_doc = doc_to_text(doc)
        tokens = token_counter(text_doc)
        chunks = num_clusters * ratio
        max_tokens = int(tokens / chunks)
        max_tokens = max(minimum_tokens, min(max_tokens, maximum_tokens))
        overlap = int(max_tokens / 10)

        splitter = TokenTextSplitter(chunk_size=max_tokens, chunk_overlap=overlap)
        split_doc = splitter.create_documents([text_doc])
        return split_doc

    def create_summarize_chain(prompt_list):
        """
        Create a langchain summarize chain from a list of prompts.

        :param prompt_list: A list containing the template, input variables, and llm to use for the chain.

        :return: A langchain summarize chain.
        """
        template = PromptTemplate(template=prompt_list[0], input_variables=([prompt_list[1]]))
        chain = load_summarize_chain(llm=prompt_list[2], chain_type='stuff', prompt=template)
        return chain

    def extract_summary_docs(langchain_document, num_clusters, find_clusters):
        """
        Automatically convert a single langchain Document object into a list of smaller langchain Document objects that represent each cluster.

        :param langchain_document: The langchain Document object to summarize.

        :param num_clusters: The number of clusters to use.

        :param find_clusters: Whether to find the optimal number of clusters to use.

        :return: A list of langchain Document objects.
        """
        split_document = split_by_tokens(langchain_document, num_clusters)
        vectors = embed_docs(split_document)

        if find_clusters:
            kmeans = kmeans_clustering(vectors, None)

        else:
            kmeans = kmeans_clustering(vectors, num_clusters)

        indices = get_closest_vectors(vectors, kmeans)
        summary_docs = map_vectors_to_docs(indices, split_document)
        return summary_docs

    def parallelize_summaries(summary_docs, initial_chain, max_workers=4):
        """
        Summarize a list of loaded langchain Document objects using multiple langchain summarize chains in parallel.

        :param summary_docs: A list of loaded langchain Document objects to summarize.

        :param initial_chain: A langchain summarize chain to use for summarization.

        :param progress_bar: A streamlit progress bar to display the progress of the summarization.

        :param max_workers: The maximum number of workers to use for parallelization.

        :return: A list of summaries.
        """
        doc_summaries = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_doc = {executor.submit(initial_chain.run, [doc]): doc.page_content for doc in summary_docs}

            for future in as_completed(future_to_doc):
                doc = future_to_doc[future]

                try:
                    summary = future.result()

                except Exception as exc:
                    print(f'{doc} generated an exception: {exc}')

                else:
                    doc_summaries.append(summary)
                    num = (len(doc_summaries)) / (len(summary_docs) + 1)
                    # progress_bar.progress(
                    #     num)  # Remove this line and all references to it if you are not using Streamlit.
        return doc_summaries

    def create_summary_from_docs(summary_docs, initial_chain, final_sum_list):
        """
        Summarize a list of loaded langchain Document objects using multiple langchain summarize chains.

        :param summary_docs: A list of loaded langchain Document objects to summarize.

        :param initial_chain: The initial langchain summarize chain to use.

        :param final_sum_list: A list containing the template, input variables, and llm to use for the final chain.

        :param use_gpt_4: Whether to use GPT-4 or GPT-3.5-turbo for summarization.

        :return: A string containing the summary.
        """

        # progress = st.progress(0)  # Create a progress bar to show the progress of summarization.
        # Remove this line and all references to it if you are not using Streamlit.

        doc_summaries = parallelize_summaries(summary_docs, initial_chain)

        summaries = '\n'.join(doc_summaries)
        count = token_counter(summaries)

        max_tokens = 3800 - int(count)

        final_sum_list[2] = get_ChatOpenAI(temperature=temperature, max_tokens=max_tokens, model_name=model_name)
        final_sum_chain = create_summarize_chain(final_sum_list)
        summaries = Document(page_content=summaries)
        final_summary = final_sum_chain.run([summaries])

        # progress.progress(1.0)  # Remove this line and all references to it if you are not using Streamlit.
        # time.sleep(0.4)  # Remove this line and all references to it if you are not using Streamlit.
        # progress.empty()  # Remove this line and all references to it if you are not using Streamlit.

        return final_summary

    async def summary_chat_iterator(model_name: str = LLM_MODEL) -> AsyncIterable[str]:
        # callback = AsyncIteratorCallbackHandler()
        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=250,
            # callbacks=[callback],
        )
        initial_prompt_list = [map_prompt, 'text', model]
        final_prompt_list = [combine_prompt, 'text', model]

        # if find_clusters:
        #     summary = doc_to_final_summary(doc, 10, initial_prompt_list, final_prompt_list, find_clusters)
        #
        # else:
        #     summary = doc_to_final_summary(doc, 10, initial_prompt_list, final_prompt_list)

        initial_prompt_list = create_summarize_chain(initial_prompt_list)
        summary_docs = extract_summary_docs(doc, 10, find_clusters)
        output = create_summary_from_docs(summary_docs, initial_prompt_list, final_prompt_list)

        yield json.dumps({"answer": output},
                         ensure_ascii=False)

        # st.markdown(summary, unsafe_allow_html=True)
        # with open(f'summaries/{name}_summary.txt', 'w') as f:
        #     f.write(summary)
        # st.text(f' Summary saved to summaries/{name}_summary.txt')

    return StreamingResponse(summary_chat_iterator(model_name=model_name),
                             media_type="text/event-stream")
