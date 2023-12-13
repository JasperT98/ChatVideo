# prompt模板使用Jinja2语法，简单点就是用双大括号代替f-string的单大括号
# 本配置文件支持热加载，修改prompt模板后无需重启服务。


# LLM对话支持的变量：
#   - input: 用户输入内容

# 知识库和搜索引擎对话支持的变量：
#   - context: 从检索结果拼接的知识文本
#   - question: 用户提出的问题
# <instruction>

    # <instruction>Tell me what user want to express or ask refer to prompt and previous context.</instruction>
    # <prompt>{{ question }}</prompt>

    # <instruction> Refer to this prompt to give me the user’s request, not prompt's answer.
    # If the user's request is not clear or there are no request, Reply to me: "***".
    # If not, Reply to me using the format: "user's request: " </instruction>
    # <prompt>{{ question }}</prompt>

    # <instruction>
    # Prompt is what user want to say.
    # Reply user based on known information in info and previous context, concisely and professionally.
    # If the answer cannot be derived from it, say "The question cannot be answered based on known information".
    # If the user's request is not clear, please ask the user to clarify the request.
    # </instruction>
    # <info>{{ context }}</info>
    # <prompt>{{ question }}</prompt>

    # Answer the user's question using the provided context and information. Keep your response concise and professional. If the available information is insufficient, state "Insufficient information to answer the question." For unclear queries, request clarification from the user.
    # </instruction>
    # <info>{{ context }}</info>
    # <prompt>{{ question }}</prompt>
    #
    # Note: Directly cite specific details from the 'info' section at the end of your response. Format your citations as follows:
    # References:
    # - citation#1: [Exact detail from 'info']
    # - citation#2: [Exact detail from 'info']
    # - citation#3: [Exact detail from 'info']

PROMPT_TEMPLATES = {
    # LLM对话模板
    "llm_chat": "{{ input }}",

    # 提取用户问题
    "question_extraction":
    """
    <instruction> 
    Refer to the user's query in the 'prompt' section below to give me the user’s request, not prompt's answer. 
    - If the user's request is unclear or there are no request, Reply to me: "***".
    - If the user's request is clear, respond using the format: "user's request: [Directly and concisely state user’s request here]" 
    </instruction>
    <prompt>{{ question }}</prompt>
    """,

    # 问题延展
    "get_detail_question":
    """
    <instruction> 
    Interpret the user's prompt in the 'prompt' section below refer to the previous conversation excerpts I provided. 
    Don't answer the user's prompt! Directly and concisely state the interpretation of the user's prompt.
    </instruction> 
    <prompt>{{ question }}</prompt>
    """,

    # 基于本地知识问答的提示词模板
    "knowledge_base_chat":
    """
    <instruction>
    Provide a response to the user's prompt in the 'prompt' section below, utilizing the details in the 'info' section below and any relevant context from previous conversation excerpts. 
    Ensure your answer is succinct and maintains a professional demeanor.
    - If the query is beyond the scope of available information, respond with: "Insufficient information to answer this query."
    - If the user's request be ambiguous or vague, request further details in a courteous manner.
    </instruction>
    <info>{{ context }}</info>
    <prompt>{{ question }}</prompt>
    """,

    # 基于agent的提示词模板
    "agent_chat":
    """
    Answer the following questions as best you can. You have access to the following offline_stat_tasks:

    {offline_stat_tasks}
    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    history:
    {history}

    Question: {input}
    Thought: {agent_scratchpad}
    """,

    "file_map":
    """
    You will be given a single section from a text. This will be enclosed in triple backticks.
    Please provide a cohesive summary of the following section excerpt, focusing on the key points and main ideas, while maintaining clarity and conciseness.
    
    '''{text}'''
    
    FULL SUMMARY:
    """,

    "file_combine":
    """
    Read all the provided summaries from a larger document. They will be enclosed in triple backticks. 
    Determine what the overall document is about and summarize it with this information in mind.
    Synthesize the info into a well-formatted easy-to-read synopsis, structured like an essay that summarizes them cohesively. 
    Do not simply reword the provided text. Do not copy the structure from the provided text.
    Avoid repetition. Connect all the ideas together.
    Preceding the synopsis, write a short, bullet form list of key takeaways.
    Format in HTML. Text should be divided into paragraphs. Paragraphs should be indented. 
    
    '''{text}'''
    
    
    """,
    "question_clustering":
    """
    You are tasked with analyzing a collection of user queries extracted from a question database, which are listed in the "Questions" section below. Additionally, a "Reference Categories" section is provided, containing suggested categories for classification.
    Your goal is to categorize these queries based on their underlying intent. Ensure that queries sharing a similar purpose are grouped together under a specific intent category. 
    Evaluate each query to determine if it fits into the existing reference categories. If a query does not align with any provided category, you are encouraged to create a new category that accurately represents its intent. Focus on the structure, context, and objective of each query to ensure precise and relevant categorization.
    
    <questions>{{ questions }}</questions>
    
    <Reference Categories>{{ Reference_Categories }}</Reference Categories>
    
    Respond with your categorization in the following structured JSON format:
    {
      "Category 1 - [Descriptive Intent Name]": {
        "1": "Sentence from Category 1",
        "2": "Another Sentence from Category 1",
        ...
      },
      "Category 2 - [Descriptive Intent Name]": {
        "1": "Sentence from Category 2",
        "2": "Another Sentence from Category 2",
        ...
      },
      ...
    }
    
    Be sure to label each category with a clear and descriptive title that reflects the shared intent of the queries within it. Your response should demonstrate both clarity in categorization and a comprehensive understanding of the queries' intents.

    
    """,


}
