import streamlit as st

from webui_pages.retrieval import retrieval_page
from webui_pages.utils import *
from streamlit_option_menu import option_menu
from webui_pages import *
import os
from configs import VERSION
from server.utils import api_address

api = ApiRequest(base_url=api_address())

if __name__ == "__main__":
    st.set_page_config(
        "ChatVideo",
        page_icon=os.path.join("img", "logo.jpg"),
        layout="centered",
        initial_sidebar_state="expanded",
    )
    if not chat_box.chat_inited:
        st.toast(
            f"Welcome ! \n\n"
            f"Now with the model '{LLM_MODEL}', you can start asking questions."
        )

    pages = {
        "Chat With A Video": {
            "icon": "chat",
            "func": dialogue_page,
        },
        "Multi Video Retrieval": {
            "icon": "chat",
            "func": retrieval_page,
        },
        "Video Library": {
            "icon": "hdd-stack",
            "func": knowledge_base_page,
        },

    }

    with st.sidebar:
        st.image(
            os.path.join(
                "img",
                "logo.jpg"
            ),
            use_column_width=True
        )
        options = list(pages)
        icons = [x["icon"] for x in pages.values()]

        default_index = 0
        selected_page = option_menu(
            "menu",
            options=options,
            icons=icons,
            menu_icon="chat-quote",
            default_index=default_index,
        )

    if selected_page in pages:
        pages[selected_page]["func"](api)
