from collections import defaultdict

import streamlit as st
from streamlit_option_menu import option_menu

from server.chat.stat import get_stat_result
from webui_pages.utils import *
from streamlit_chatbox import *
from datetime import datetime
from server.chat.search_engine_chat import SEARCH_ENGINES
import os
from configs import LLM_MODEL, TEMPERATURE
from server.utils import get_model_worker_config
from typing import List, Dict

import aiofiles

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


def dialogue_page(api: ApiRequest):
    chat_box.init_session()

    st.session_state["has_video"] = False
    st.session_state["video_url"] = None
    st.session_state["video_start_time"] = 0
    st.session_state["stat"] = "Please generate a stat report first"

    with st.sidebar:
        # dialogue_mode = st.selectbox("Please select the conversation mode：",
        #                              [
        #                                  # "LLM 对话",
        #                                  "Q/A",
        #                                  "Search Engine Q&A",
        #                                  # "自定义Agent问答",
        #                              ],
        #                              index=1,
        #                              on_change=on_mode_change,
        #                              key="dialogue_mode",
        #                              )
        dialogue_mode = option_menu("", ["Q/A", "Search Engine Q&A"],
                                    icons=['house', 'cloud'],
                                    menu_icon="cloud-upload", default_index=0, orientation="horizontal")

        setting_section, video_section, stat_section = st.tabs(["Setting", "Video", "Stat"])

        # TODO: 对话模型与会话绑定
        def on_mode_change():
            mode = st.session_state.dialogue_mode
            text = f"Switched to {mode} mode."
            if mode == "Q/A":
                cur_kb = st.session_state.get("selected_kb")
                if cur_kb:
                    text = f"{text} current video： `{cur_kb}`。"
            st.toast(text)
            # sac.alert(text, description="descp", type="success", closable=True, banner=True)

        def on_llm_change():
            config = get_model_worker_config(llm_model)
            if not config.get("online_api"):  # 只有本地model_worker可以切换模型
                st.session_state["prev_llm_model"] = llm_model
            st.session_state["cur_llm_model"] = st.session_state.llm_model

        def llm_model_format_func(x):
            if x in running_models:
                return f"{x} (Running)"
            return x

        def on_kb_change():
            st.toast(f"video loaded： {st.session_state.selected_kb}")

        with setting_section:


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

            if dialogue_mode == "Q/A":
                with st.expander("Scope configuration", True):
                    kb_top_k = st.number_input("Number of matches for reference: ", 1, 20, VECTOR_SEARCH_TOP_K)

                    ## Bge 模型会超过1
                    score_threshold = st.slider("Knowledge matching score threshold: ", 0.0, 1.0, float(SCORE_THRESHOLD),
                                                0.01)

                    # chunk_content = st.checkbox("关联上下文", False, disabled=True)
                    # chunk_size = st.slider("关联长度：", 0, 500, 250, disabled=True)

            elif dialogue_mode == "Search Engine Q&A":
                search_engine_list = list(SEARCH_ENGINES.keys())
                with st.expander("Search Engine Configuration", True):
                    search_engine = st.selectbox(
                        label="Please select a search engine",
                        options=search_engine_list,
                        index=search_engine_list.index("duckduckgo") if "duckduckgo" in search_engine_list else 0,
                    )
                    se_top_k = st.number_input("Number of matches for reference：", 1, 20, SEARCH_ENGINE_TOP_K)

        with video_section:
            kb_list = api.list_knowledge_bases(no_remote_api=True)
            selected_kb = st.selectbox(
                "Please select a video: ",
                kb_list,
                on_change=on_kb_change,
                key="selected_kb",
            )
            video_list = api.list_video_path(knowledge_base_name=selected_kb)
            if video_list and len(video_list) > 0:
                st.session_state.video_url = video_list[0]
                st.session_state.video_start_time = 0
                st.session_state.has_video = True

        with stat_section:
            if st.button("Generate statistics"):
                # 调用函数处理输入
                st.session_state.stat = get_stat_result(selected_kb)

            # 在页面上显示结果
            st.write(st.session_state.stat)

    # Display chat messages from history on app rerun
    chat_box.output_messages()

    chat_input_placeholder = "Please Enter the content of the conversation, use Shift Enter for line breaks"

    if prompt := st.chat_input(chat_input_placeholder, key="prompt"):
        history = get_messages_history(history_len)
        chat_box.user_say(prompt)
        if dialogue_mode == "LLM 对话":
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
        elif dialogue_mode == "自定义Agent问答":
            chat_box.ai_say([
                f"Thinking...", ])
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
                    chat_box.update_msg("\n\n".join(d.get("offline_stat_tasks", [])), element_index=element_index,
                                        streaming=False)
            chat_box.update_msg(text, element_index=0, streaming=False)
        elif dialogue_mode == "Q/A":
            chat_box.ai_say([
                f"Thinking...",
                Markdown("...", in_expander=True, title="matching result", state="complete"),
            ])
            text = ""

            # text_ = ""
            # r = api.chat_chat(prompt, history=history, model=llm_model, prompt_name="question_extend", temperature=temperature)
            # for t in r:
            #     if error_msg := check_error_msg(t):  # check whether error occured
            #         st.error(error_msg)
            #         break
            #     text_ += t
            #
            # logger.info(f"======= question_extend: {text_} =========")

            for d in api.knowledge_base_chat(prompt,
                                             knowledge_base_name=selected_kb,
                                             top_k=kb_top_k,
                                             score_threshold=score_threshold,
                                             history=history,
                                             model=llm_model,
                                             prompt_name="knowledge_base_chat",
                                             temperature=temperature):
                if error_msg := check_error_msg(d):  # check whether error occured
                    st.error(error_msg)
                elif chunk := d.get("answer"):
                    text += chunk
                    chat_box.update_msg(text, element_index=0)
            chat_box.update_msg(text, element_index=0, streaming=False)
            chat_box.update_msg("\n\n".join(d.get("docs", [])), element_index=1, streaming=False)
            # 使用record_question保存问题
        elif dialogue_mode == "Search Engine Q&A":
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
        file_name=f"{now:%Y-%m-%d %H.%M}_对话记录.md",
        mime="text/markdown",
        use_container_width=True,
    )

    if dialogue_mode == "Q/A":
        with st.expander("Generate from this video", True):
            cols = st.columns(3)

            if cols[0].button(
                    "Summarize",
                    use_container_width=True,
            ):
                chat_box.ai_say("Here is the summary of this video: \n\n ")

                summary = ""
                # TODO：给 summary 赋值
                for d in api.summary_chat(knowledge_base_name=selected_kb,
                                          model=llm_model,
                                          temperature=temperature):
                    if error_msg := check_error_msg(d):  # check whether error occured
                        st.error(error_msg)
                    elif chunk := d.get("answer"):
                        summary += chunk
                        chat_box.update_msg(summary, element_index=0)

                chat_box.update_msg(summary, streaming=False)

                # 有这个，内容就会在框外展示。没有，内容就会在窗内展示
                st.experimental_rerun()

            # if cols[1].button(
            #         "Outline",
            #         use_container_width=True,
            # ):
            #     outline = ""
            #     # TODO：给 outline 赋值
            #
            #     # 展示 summary
            #     chat_box.ai_say("Here is the outline of this video: \n\n " + outline)
            #
            #     # 有这个，内容就会在框外展示。没有，内容就会在窗内展示
            #     st.experimental_rerun()
            #
            # if cols[2].button(
            #         "Find action items",
            #         use_container_width=True,
            # ):
            #     action_items = ""
            #     # TODO：给 action items 赋值
            #
            #     # 展示 action items
            #     chat_box.ai_say("Here is all action items of this video: \n\n ")
            #
            #     # 有这个，内容就会在框外展示。没有，内容就会在窗内展示
            #     st.experimental_rerun()

    if st.session_state.has_video:
        video_section.video(st.session_state.video_url, start_time=st.session_state.video_start_time)
        video_section.link_button("Go to Youtube", st.session_state.video_url+"&t="+str(st.session_state.video_start_time)+"s")
        video_section.divider()
        # st.video(video_url, start_time=time)
        # st.divider()
    else:
        video_section.markdown("No video uploaded, can just chat with the video")