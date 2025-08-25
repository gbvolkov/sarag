import os
#os.environ["TRANSFORMERS_VERBOSITY"] = "info"

os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

import config

from pprint import pprint

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt.chat_agent_executor import create_react_agent

from langchain_gigachat import GigaChat

from retrievers.retriever import get_search_tool

from tasks_info_tool import tasks_info


search_kb = get_search_tool()

search_tools = [
    search_kb,
    tasks_info,
]

with open("prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()    

"""
llm = GigaChat(
    credentials=config.GIGA_CHAT_AUTH, 
    model="GigaChat-2",
    verify_ssl_certs=False,
    temperature=0,
    frequency_penalty=0,
    scope = config.GIGA_CHAT_SCOPE)
"""
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

memory = MemorySaver()
agent =  create_react_agent(
    model=llm, 
    tools=search_tools, 
    prompt=system_prompt, 
    name="assistant_sa", 
    #post_model_hook=get_validator("sd_agent"),
    checkpointer=memory, 
    debug=config.DEBUG_WORKFLOW)

