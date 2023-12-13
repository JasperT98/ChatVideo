import streamlit as st
from webui_pages.utils import *
from streamlit_chatbox import *
from datetime import datetime
from server.chat.search_engine_chat import SEARCH_ENGINES
import os
from configs import LLM_MODEL, TEMPERATURE
from server.utils import get_model_worker_config
from typing import List, Dict

chat_box = ChatBox(
    assistant_avatar=os.path.join(
        "img",
        "Robot.png"
    )
)


def get_messages_history(history_len: int, content_in_expander: bool = False) -> List[Dict]:
    '''
    返回消息历史。
    content_in_expander控制是否返回expander元素中的内容，一般导出的时候可以选上，传入LLM的history不需要
    '''

    def filter(msg):
        content = [x for x in msg["elements"] if x._output_method in ["markdown", "text"]]
        if not content_in_expander:
            content = [x for x in content if not x._in_expander]
        content = [x.content for x in content]

        return {
            "role": msg["role"],
            "content": "\n\n".join(content),
        }

    return chat_box.filter_history(history_len=history_len, filter=filter)


def retrieval_page(api: ApiRequest):
    chat_box.init_session()

    with st.sidebar:
        # TODO: 对话模型与会话绑定
        def on_mode_change():
            mode = st.session_state.retrieval_mode
            text = f"Switched to {mode} mode."
            if mode == "Q/A":
                cur_kb = st.session_state.get("selected_kbs")
                if cur_kb:
                    text = f"{text} current video： `{cur_kb}`。"
            st.toast(text)
            # sac.alert(text, description="descp", type="success", closable=True, banner=True)

        retrieval_mode = st.selectbox("Please select a retrieval mode：",
                                     ["Brief Retrieval",
                                      "Detail Retrieval",
                                      ],
                                     index=1,
                                     on_change=on_mode_change,
                                     key="retrieval_mode",
                                     )

        def on_llm_change():
            config = get_model_worker_config(llm_model)
            if not config.get("online_api"):  # 只有本地model_worker可以切换模型
                st.session_state["prev_llm_model"] = llm_model
            st.session_state["cur_llm_model"] = st.session_state.llm_model

        def llm_model_format_func(x):
            if x in running_models:
                return f"{x} (Running)"
            return x

        running_models = api.list_running_models()
        available_models = []
        config_models = api.list_config_models()
        for models in config_models.values():
            for m in models:
                if m not in running_models:
                    available_models.append(m)
        llm_models = running_models + available_models
        index = llm_models.index(st.session_state.get("cur_llm_model", LLM_MODEL))
        # llm_model = st.selectbox("选择LLM模型：",
        #                          llm_models,
        #                          index,
        #                          format_func=llm_model_format_func,
        #                          on_change=on_llm_change,
        #                          key="llm_model",
        #                          )
        llm_model = LLM_MODEL
        if (st.session_state.get("prev_llm_model") != llm_model
                and not get_model_worker_config(llm_model).get("online_api")
                and llm_model not in running_models):
            with st.spinner(f"Loading model： {llm_model}，do not perform operations or refresh the page"):
                prev_model = st.session_state.get("prev_llm_model")
                r = api.change_llm_model(prev_model, llm_model)
                if msg := check_error_msg(r):
                    st.error(msg)
                elif msg := check_success_msg(r):
                    st.success(msg)
                    st.session_state["prev_llm_model"] = llm_model

        temperature = st.slider("Temperature：", 0.0, 1.0, TEMPERATURE, 0.01)

        ## 部分模型可以超过10抡对话
        history_len = st.number_input("Number of historical dialogues：", 0, 20, HISTORY_LEN)

        def on_kb_change():
            st.toast(f"video loaded： {st.session_state.selected_kbs}")

        if retrieval_mode == "Detail Retrieval":
            with st.expander("Scope configuration", True):
                kb_list = api.list_knowledge_bases(no_remote_api=True)
                selected_kbs = st.multiselect(
                    "Please select videos:",
                    kb_list,
                    on_change=on_kb_change,
                    key="selected_kbs",
                )
                kb_top_k = st.number_input("Number of matches for reference: ", 1, 20, VECTOR_SEARCH_TOP_K)

                ## Bge 模型会超过1
                score_threshold = st.slider("Knowledge matching score threshold: ", 0.0, 1.0, float(SCORE_THRESHOLD), 0.01)

                # chunk_content = st.checkbox("关联上下文", False, disabled=True)
                # chunk_size = st.slider("关联长度：", 0, 500, 250, disabled=True)
        elif retrieval_mode == "搜索引擎问答":
            search_engine_list = list(SEARCH_ENGINES.keys())
            with st.expander("搜索引擎配置", True):
                search_engine = st.selectbox(
                    label="请选择搜索引擎",
                    options=search_engine_list,
                    index=search_engine_list.index("duckduckgo") if "duckduckgo" in search_engine_list else 0,
                )
                se_top_k = st.number_input("匹配搜索结果条数：", 1, 20, SEARCH_ENGINE_TOP_K)

    # Display chat messages from history on app rerun

    chat_box.output_messages()

    chat_input_placeholder = "Please Enter the content of the conversation, use Shift Enter for line breaks"

    if prompt := st.chat_input(chat_input_placeholder, key="prompt"):
        history = get_messages_history(history_len)
        chat_box.user_say(prompt)
        if retrieval_mode == "LLM 对话":
            chat_box.ai_say("Thinking...")
            text = ""
            r = api.chat_chat(prompt, history=history, model=llm_model, temperature=temperature)
            for t in r:
                if error_msg := check_error_msg(t):  # check whether error occured
                    st.error(error_msg)
                    break
                text += t
                chat_box.update_msg(text)
            chat_box.update_msg(text, streaming=False)  # 更新最终的字符串，去除光标


        elif retrieval_mode == "自定义Agent问答":
            chat_box.ai_say([
                f"Thinking...",])
            text = ""
            element_index = 0
            for d in api.agent_chat(prompt,
                                    history=history,
                                    model=llm_model,
                                    temperature=temperature):
                try:
                    d = json.loads(d)
                except:
                    pass
                if error_msg := check_error_msg(d):  # check whether error occured
                    st.error(error_msg)

                elif chunk := d.get("answer"):
                    text += chunk
                    chat_box.update_msg(text, element_index=0)
                elif chunk := d.get("offline_stat_tasks"):
                    element_index += 1
                    chat_box.insert_msg(Markdown("...", in_expander=True, title="使用工具...", state="complete"))
                    chat_box.update_msg("\n\n".join(d.get("offline_stat_tasks", [])), element_index=element_index, streaming=False)
            chat_box.update_msg(text, element_index=0, streaming=False)
        elif retrieval_mode == "Detail Retrieval":
            # chat_box.ai_say([
            #     f"Thinking...",
            #     Markdown("...", in_expander=True, title="matching result", state="complete"),
            # ])
            text = ""
            # 可以开多线程
            for selected_kb in selected_kbs:
                for d in api.knowledge_base_chat(prompt,
                                                 knowledge_base_name=selected_kb,
                                                 top_k=kb_top_k,
                                                 score_threshold=score_threshold,
                                                 history=history,
                                                 model=llm_model,
                                                 temperature=temperature):
                    if error_msg := check_error_msg(d):  # check whether error occured
                        st.error(error_msg)
                    elif chunk := d.get("answer"):
                        text += chunk
                        # chat_box.update_msg(text, element_index=0)
                # chat_box.update_msg(text, element_index=0, streaming=False)
                # chat_box.update_msg("\n\n".join(d.get("docs", [])), element_index=1, streaming=False)
                chat_box.ai_say([
                    text,
                    Markdown("\n\n".join(d.get("docs", [])), in_expander=True, title="matching result", state="complete"),
                ])
        elif retrieval_mode == "搜索引擎问答":
            chat_box.ai_say([
                f"正在执行 `{search_engine}` 搜索...",
                Markdown("...", in_expander=True, title="网络搜索结果", state="complete"),
            ])
            text = ""
            for d in api.search_engine_chat(prompt,
                                            search_engine_name=search_engine,
                                            top_k=se_top_k,
                                            history=history,
                                            model=llm_model,
                                            temperature=temperature):
                if error_msg := check_error_msg(d):  # check whether error occured
                    st.error(error_msg)
                elif chunk := d.get("answer"):
                    text += chunk
                    chat_box.update_msg(text, element_index=0)
            chat_box.update_msg(text, element_index=0, streaming=False)
            chat_box.update_msg("\n\n".join(d.get("docs", [])), element_index=1, streaming=False)

    now = datetime.now()
    with st.sidebar:

        cols = st.columns(2)
        export_btn = cols[0]
        if cols[1].button(
                "Start a new topic",
                use_container_width=True,
        ):
            chat_box.reset_history()
            st.experimental_rerun()

    export_btn.download_button(
        "Export dialogue",
        "".join(chat_box.export2md()),
        file_name=f"{now:%Y-%m-%d %H.%M}_record.md",
        mime="text/markdown",
        use_container_width=True,
    )
