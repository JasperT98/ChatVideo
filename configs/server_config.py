import sys
from configs.model_config import LLM_DEVICE

# httpx 请求默认超时时间（秒）。如果加载模型或对话较慢，出现超时错误，可以适当加大该值。
HTTPX_DEFAULT_TIMEOUT = 300.0

# API 是否开启跨域，默认为False，如果需要开启，请设置为True
# is open cross domain
OPEN_CROSS_DOMAIN = False

# 各服务器默认绑定host。如改为"0.0.0.0"需要修改下方所有XX_SERVER的host
DEFAULT_BIND_HOST = "0.0.0.0"

# webui.py server
WEBUI_SERVER = {
    "host": DEFAULT_BIND_HOST,
    "port": 8501,
}

# api.py server
API_SERVER = {
    "host": DEFAULT_BIND_HOST,
    "port": 7861,
}

# fastchat openai_api server
FSCHAT_OPENAI_API = {
    "host": DEFAULT_BIND_HOST,
    "port": 20000,
}

# fastchat model_worker server
# 这些模型必须是在model_config.MODEL_PATH或ONLINE_MODEL中正确配置的。
# 在启动startup.py时，可用通过`--model-worker --model-name xxxx`指定模型，不指定则为LLM_MODEL
FSCHAT_MODEL_WORKERS = {
    # 所有模型共用的默认配置，可在模型专项配置中进行覆盖。
    "default": {
        "host": DEFAULT_BIND_HOST,
        "port": 20002,
        "device": LLM_DEVICE,
        # False,'vllm',使用的推理加速框架,使用vllm如果出现HuggingFace通信问题，参见doc/FAQ
        "infer_turbo": "vllm" if sys.platform.startswith("linux") else False,


    },

}

# fastchat multi model worker server
FSCHAT_MULTI_MODEL_WORKERS = {
    # TODO:
}

# fastchat controller server
FSCHAT_CONTROLLER = {
    "host": DEFAULT_BIND_HOST,
    "port": 20001,
    "dispatch_method": "shortest_queue",
}
