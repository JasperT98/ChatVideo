import string
from datetime import datetime
import random
from io import StringIO

import streamlit as st
from streamlit_option_menu import option_menu
from youtube_transcript_api import YouTubeTranscriptApi

from server.chat import summary_chat
from webui_pages.utils import *
from st_aggrid import AgGrid, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import pandas as pd
from server.knowledge_base.utils import get_file_path, LOADER_DICT
from server.knowledge_base.kb_service.base import get_kb_details, get_kb_file_details
from typing import Literal, Dict, Tuple
from configs import (kbs_config,
                     EMBEDDING_MODEL, DEFAULT_VS_TYPE,
                     CHUNK_SIZE, OVERLAP_SIZE, ZH_TITLE_ENHANCE, LLM_MODEL, TEMPERATURE)
from server.utils import list_embed_models
import os
import time

# SENTENCE_SIZE = 100

cell_renderer = JsCode("""function(params) {if(params.value==true){return '✓'}else{return '×'}}""")


def is_youtube_url(url):
    """
    Function to check if a given string is a valid YouTube URL.

    Args:
    url (str): The string to be checked.

    Returns:
    bool: True if the string is a YouTube URL, False otherwise.
    """

    # YouTube URL patterns
    youtube_patterns = [
        'youtube.com/watch?v=',  # Standard URL format
    ]

    # Check if any of the patterns exist in the URL
    return any(pattern in url for pattern in youtube_patterns)


def config_aggrid(
        df: pd.DataFrame,
        columns: Dict[Tuple[str, str], Dict] = {},
        selection_mode: Literal["single", "multiple", "disabled"] = "single",
        use_checkbox: bool = False,
) -> GridOptionsBuilder:
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_column("No", width=40)
    for (col, header), kw in columns.items():
        gb.configure_column(col, header, wrapHeaderText=True, **kw)
    gb.configure_selection(
        selection_mode=selection_mode,
        use_checkbox=use_checkbox,
        # pre_selected_rows=st.session_state.get("selected_rows", [0]),
    )
    return gb


def file_exists(kb: str, selected_rows: List) -> Tuple[str, str]:
    '''
    check whether a doc file exists in local knowledge base folder.
    return the file's name and path if it exists.
    '''
    if selected_rows:
        file_name = selected_rows[0]["file_name"]
        file_path = get_file_path(kb, file_name)
        if os.path.isfile(file_path):
            return file_name, file_path
    return "", ""


async def test1():
    logger.info("============== test1 start ==============")
    await asyncio.sleep(3)
    logger.info("============== test1 done ==============")


async def test2():
    logger.info("============== test2 start ==============")
    await asyncio.sleep(5)
    logger.info("============== test2 done ==============")


async def test3():
    logger.info("============== test3 start ==============")
    await asyncio.sleep(10)
    logger.info("============== test3 done ==============")


async def async_generate_summary_and_upload_kb_docs(api: ApiRequest,
                                                    file: List[Union[str, Path, bytes]],
                                                    knowledge_base_name: str,
                                                    model=LLM_MODEL,
                                                    temperature=TEMPERATURE
                                                    ):
    logger.info("============== generate summary start ==============")
    # TODO: 生成summary
    summary = ""
    for d in api.summary_chat(knowledge_base_name=knowledge_base_name,
                              file=file,
                              model=model,
                              temperature=temperature):
        if error_msg := check_error_msg(d):  # check whether error occured
            st.error(error_msg)
        elif chunk := d.get("answer"):
            summary += chunk
    logger.info("SUMMARY!!!!!: " + summary)
    #  TODO: 保存summary，并上传
    logger.info("============== upload summary start ==============")

    logger.info("============== upload summary end ==============")


def rename_file(original_name, insert_text):
    # 分离文件的基本名和扩展名
    name, ext = os.path.splitext(original_name)
    # 在基本名和扩展名之间插入指定的文本
    new_name = f"{name}_{insert_text}{ext}"
    return new_name


def append_timestamp(name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
    return name + "_" + timestamp + "_" + random_str


def knowledge_base_page(api: ApiRequest):
    try:
        kb_list = {x["kb_name"]: x for x in get_kb_details()}
    except Exception as e:
        st.error(
            "获取知识库信息错误，请检查是否已按照 `README.md` 中 `4 知识库初始化与迁移` 步骤完成初始化或迁移，或是否为数据库连接错误。")
        st.stop()
    kb_names = list(kb_list.keys())

    if "selected_kb_name" in st.session_state and st.session_state["selected_kb_name"] in kb_names:
        selected_kb_index = kb_names.index(st.session_state["selected_kb_name"])
    else:
        selected_kb_index = 0

    def format_selected_kb(kb_name: str) -> str:
        if kb := kb_list.get(kb_name):
            return f"{kb_name} ({kb['vs_type']} @ {kb['embed_model']})"
        else:
            return kb_name

    selected_kb = st.selectbox(
        "Please select or create a video knowledge base: ",
        kb_names + ["Create video knowledge base"],
        format_func=format_selected_kb,
        index=selected_kb_index
    )

    if selected_kb == "Create video knowledge base":
        with st.form("Create video knowledge base"):

            kb_name = st.text_input(
                "New video knowledge base name",
                placeholder="New video knowledge base name, Chinese naming not supported",
                key="kb_name",
            )

            cols = st.columns(2)

            vs_types = list(kbs_config.keys())
            vs_type = cols[0].selectbox(
                "Vector database type",
                vs_types,
                index=vs_types.index(DEFAULT_VS_TYPE),
                key="vs_type",
            )

            embed_models = list_embed_models()

            embed_model = cols[1].selectbox(
                "Embedding model",
                embed_models,
                index=embed_models.index(EMBEDDING_MODEL),
                key="embed_model",
            )

            submit_create_kb = st.form_submit_button(
                "Build",
                # disabled=not bool(kb_name),
                use_container_width=True,
            )

        if submit_create_kb:
            if not kb_name or not kb_name.strip():
                st.error(f"The knowledge base name cannot be empty!")
            elif kb_name in kb_list:
                st.error(f"The knowledge base named {kb_name} already exists!")
            else:
                ret = api.create_knowledge_base(
                    knowledge_base_name=kb_name,
                    vector_store_type=vs_type,
                    embed_model=embed_model,
                )
                st.toast(ret.get("msg", " "))
                st.session_state["selected_kb_name"] = kb_name
                st.experimental_rerun()

    elif selected_kb:
        kb = selected_kb

        # selected_content = st.selectbox(
        #     "Please select the content: ",
        #     ["Transcript", "Background info", "Summary"],
        #     key="selected_content",
        # )
        # selected_content = option_menu("Select the content",
        #                                ["Transcript", "Background info", "Summary"],
        #                                default_index=0,
        #                                key="selected_content",)
        st.divider()
        st.write("Please select content type：")
        selected_content = option_menu("", ["Transcript", "Background info", "Summary"],
                                       icons=['list-task', 'list-task', "list-task"],
                                       menu_icon="cloud-upload", default_index=0, orientation="horizontal")

        generate_and_update_summary = False
        # 上传的文件（一个BytesIO对象）
        files = []
        link = None

        if selected_content == "Transcript":
            # input_method = st.selectbox("Source", ('Document', 'YouTube URL'))

            input_method = option_menu("", ["Document", "YouTube URL"],
                                           icons=['house', 'cloud'],
                                           menu_icon="cloud-upload", default_index=0, orientation="horizontal")

            if input_method == 'Document':
                # 上传文件（只允许传一个）
                file = st.file_uploader("Upload file：",
                                        [i for ls in LOADER_DICT.values() for i in ls],
                                        accept_multiple_files=False,
                                        )
                # 改文件名：transcript.xxx
                if file:
                    _, ext = os.path.splitext(file.name)
                    file.name = f"transcript{ext}"
                    # file.name = rename_file(file.name, "transcript")
                    files.append(file)
            if input_method == 'YouTube URL':
                youtube_url = st.text_input("Enter a YouTube URL")
                if youtube_url:
                    if not is_youtube_url(youtube_url):
                        st.error("Please enter a valid YouTube URL.")
                    else:
                        link = ("transcript", youtube_url)
            # with st.sidebar:
            with st.expander(
                    "File processing configuration",
                    expanded=True,
            ):
                cols = st.columns(3)
                chunk_size = cols[0].number_input("Maximum length of a single text segment:", 1, 1000, CHUNK_SIZE)
                chunk_overlap = cols[1].number_input("Length of overlapping adjacent texts:", 0, chunk_size,
                                                     OVERLAP_SIZE)
                cols[2].write("")
                cols[2].write("")
                zh_title_enhance = cols[2].toggle("Enable title enhancement", ZH_TITLE_ENHANCE)
                generate_and_update_summary = cols[2].toggle("Generate and Update summary")

        elif selected_content == "Background info":
            # 上传文件（允许传多个）
            file = st.file_uploader("Upload file：",
                                    [i for ls in LOADER_DICT.values() for i in ls],
                                    accept_multiple_files=True,
                                    )
            if file:
                for f in file:
                    # 改文件名：background_info_xxx.xxx
                    _, ext = os.path.splitext(f.name)
                    f.name = f"{append_timestamp('background_info')}{ext}"
                    # file.name = rename_file(file.name, append_timestamp("background_info"))
                    files.append(f)
            # with st.sidebar:
            with st.expander(
                    "File processing configuration",
                    expanded=True,
            ):
                cols = st.columns(3)
                chunk_size = cols[0].number_input("Maximum length of a single text segment:", 1, 1000, CHUNK_SIZE)
                chunk_overlap = cols[1].number_input("Length of overlapping adjacent texts:", 0, chunk_size,
                                                     OVERLAP_SIZE)
                cols[2].write("")
                cols[2].write("")
                zh_title_enhance = cols[2].toggle("Enable title enhancement", ZH_TITLE_ENHANCE)

        elif selected_content == "Summary":
            # 上传文件（只允许传一个）
            file = st.file_uploader("Upload file：",
                                    [i for ls in LOADER_DICT.values() for i in ls],
                                    accept_multiple_files=False,
                                    )
            if file:
                # 改文件名：summary.txt
                _, ext = os.path.splitext(file.name)
                file.name = f"transcript{ext}"
                # file.name = rename_file(file.name, "Summary")
                files.append(file)

            # with st.sidebar:
            with st.expander(
                    "File processing configuration",
                    expanded=True,
            ):
                cols = st.columns(3)
                chunk_size = cols[0].number_input("Maximum length of a single text segment:", 1, 1000, CHUNK_SIZE)
                chunk_overlap = cols[1].number_input("Length of overlapping adjacent texts:", 0, chunk_size,
                                                     OVERLAP_SIZE)
                cols[2].write("")
                cols[2].write("")
                zh_title_enhance = cols[2].toggle("Enable title enhancement", ZH_TITLE_ENHANCE)

        if st.button(
                "Add files to the knowledge base",
                # use_container_width=True,
                disabled=(files == [] or files == [None] or files == [()]) and (
                        link is None or link == ()),
        ):
            # handle_summary = asyncio.create_task(summary_chat())
            if generate_and_update_summary:
                logger.info("=================== summary + upload =================")
                results = asyncio.run(async_run_multi_func(
                    api.upload_kb_docs(knowledge_base_name=kb,
                                       files=files,
                                       link=link,
                                       override=True,
                                       chunk_size=chunk_size,
                                       chunk_overlap=chunk_overlap,
                                       zh_title_enhance=zh_title_enhance),
                    test1(),
                    test2(),
                    test3(),
                    async_generate_summary_and_upload_kb_docs(api,
                                                              file=files,
                                                              knowledge_base_name=kb),
                ))

                for result in results:
                    logger.info(f"得到执行结果: {result}")
            else:
                logger.info("=================== upload =================")
                results = asyncio.run(async_run_multi_func(api.upload_kb_docs(knowledge_base_name=kb,
                                                                              files=files,
                                                                              link=link,
                                                                              override=True,
                                                                              chunk_size=chunk_size,
                                                                              chunk_overlap=chunk_overlap,
                                                                              zh_title_enhance=zh_title_enhance,
                                                                              )))
                for result in results:
                    logger.info(f"得到执行结果: {result}")
                # if msg := check_success_msg(ret):
                #     st.toast(msg, icon="✔")
                # elif msg := check_error_msg(ret):
                #     st.toast(msg, icon="✖")

        st.divider()

        # 知识库详情
        # st.info("请选择文件，点击按钮进行操作。")
        doc_details = pd.DataFrame(get_kb_file_details(kb))
        if not len(doc_details):
            st.info(f"There are no files in the knowledge base `{kb}`")
        else:
            st.write(f"Files already exist in the knowledge base `{kb}`：")
            st.info(
                "The knowledge base contains source files and vector libraries. Please select a file from the table below to proceed with the operation.")
            doc_details.drop(columns=["kb_name"], inplace=True)
            doc_details = doc_details[[
                "No", "file_name", "document_loader", "text_splitter", "docs_count", "in_folder", "in_db",
            ]]
            # doc_details["in_folder"] = doc_details["in_folder"].replace(True, "✓").replace(False, "×")
            # doc_details["in_db"] = doc_details["in_db"].replace(True, "✓").replace(False, "×")
            gb = config_aggrid(
                doc_details,
                {
                    ("No", "No"): {},
                    ("file_name", "file name"): {},
                    # ("file_ext", "文档类型"): {},
                    # ("file_version", "文档版本"): {},
                    ("document_loader", "document loader"): {},
                    ("docs_count", "docs count"): {},
                    ("text_splitter", "textsplitter"): {},
                    # ("create_time", "创建时间"): {},
                    ("in_folder", "source file"): {"cellRenderer": cell_renderer},
                    ("in_db", "vector database"): {"cellRenderer": cell_renderer},
                },
                "multiple",
            )

            doc_grid = AgGrid(
                doc_details,
                gb.build(),
                columns_auto_size_mode="FIT_CONTENTS",
                theme="alpine",
                custom_css={
                    "#gridToolBar": {"display": "none"},
                },
                allow_unsafe_jscode=True,
                enable_enterprise_modules=False
            )

            selected_rows = doc_grid.get("selected_rows", [])

            cols = st.columns(4)
            file_name, file_path = file_exists(kb, selected_rows)
            if file_path:
                with open(file_path, "rb") as fp:
                    cols[0].download_button(
                        "Download selected document",
                        fp,
                        file_name=file_name,
                        use_container_width=True, )
            else:
                cols[0].download_button(
                    "Download selected document",
                    "",
                    disabled=True,
                    use_container_width=True, )

            st.write()
            # 将文件分词并加载到向量库中
            if cols[1].button(
                    "Re-add to vector database" if selected_rows and (
                            pd.DataFrame(selected_rows)["in_db"]).any() else "Add to vector database",
                    disabled=not file_exists(kb, selected_rows)[0],
                    use_container_width=True,
            ):
                file_names = [row["file_name"] for row in selected_rows]
                api.update_kb_docs(kb,
                                   file_names=file_names,
                                   chunk_size=chunk_size,
                                   chunk_overlap=chunk_overlap,
                                   zh_title_enhance=zh_title_enhance)
                st.experimental_rerun()

            # 将文件从向量库中删除，但不删除文件本身。
            if cols[2].button(
                    "Delete (from vector database)",
                    disabled=not (selected_rows and selected_rows[0]["in_db"]),
                    use_container_width=True,
            ):
                file_names = [row["file_name"] for row in selected_rows]
                api.delete_kb_docs(kb, file_names=file_names)
                st.experimental_rerun()

            if cols[3].button(
                    "Delete (from the knowledge base)",
                    type="primary",
                    use_container_width=True,
            ):
                file_names = [row["file_name"] for row in selected_rows]
                api.delete_kb_docs(kb, file_names=file_names, delete_content=True)
                st.experimental_rerun()

        st.divider()

        cols = st.columns(2)

        if cols[0].button(
                "Rebuild the vector database from the source file",
                # help="无需上传文件，通过其它方式将文档拷贝到对应知识库content目录下，点击本按钮即可重建知识库。",
                use_container_width=True,
                type="primary",
        ):
            with st.spinner(
                    "During vector database reconstruction, please wait patiently, do not refresh or close the page."):
                empty = st.empty()
                empty.progress(0.0, "")
                for d in api.recreate_vector_store(kb,
                                                   chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap,
                                                   zh_title_enhance=zh_title_enhance):
                    if msg := check_error_msg(d):
                        st.toast(msg)
                    else:
                        empty.progress(d["finished"] / d["total"], d["msg"])
                st.experimental_rerun()

        if cols[1].button(
                "Delete a knowledge base",
                use_container_width=True,
        ):
            ret = api.delete_knowledge_base(kb)
            st.toast(ret.get("msg", " "))
            time.sleep(1)
            st.experimental_rerun()
