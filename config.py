from dotenv import load_dotenv,dotenv_values
import os

from pathlib import Path


if os.path.exists("gv.env"):
    load_dotenv('./gv.env')
else:
    documents_path = Path.home() / ".env"
    load_dotenv(os.path.join(documents_path, 'gv.env'))

CHROME_DRIVER_PATH = './bin/chromedriver.exe'
LOGIN_URL = 'https://kb.ileasing.ru/auth/sign-in?redirect=%2Fspace%2Fa100dc8d-3af0-418c-8634-f09f1fdb06f2%2Farticle%2Fa1038bbc-e5d9-4b5a-9482-2739c19cb6cb'
TARGET_URL = 'https://kb.ileasing.ru/space/a100dc8d-3af0-418c-8634-f09f1fdb06f2/article/a1038bbc-e5d9-4b5a-9482-2739c19cb6cb'
USERNAME = '7810155'
PASSWORD = os.environ.get('IL_PWD') 

LOCAL_MODEL_NAME='/models/Meta-Llama-3.1-8B-Instruct-Q8_0'

TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_ASSISTANT_BOT_TOKEN')
#EMBEDDING_MODEL='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

#EMBEDDING_MODEL='sentence-transformers/distiluse-base-multilingual-cased-v1'
EMBEDDING_MODEL=os.environ.get('EMBEDDING_MODEL') or '/models/multilingual-e5-large'
#EMBEDDING_MODEL='jinaai/jina-embeddings-v3'

GIGA_CHAT_USER_ID=os.environ.get('GIGA_CHAT_USER_ID')
GIGA_CHAT_SECRET = os.environ.get('GIGA_CHAT_SECRET')
GIGA_CHAT_AUTH = os.environ.get('GIGA_CHAT_AUTH')
GIGA_CHAT_SCOPE = "GIGACHAT_API_PERS"

LANGCHAIN_API_KEY = os.environ.get('LANGCHAIN_API_KEY')
#LANGCHAIN_API_KEY_DEV = os.environ.get('LANGCHAIN_API_KEY_DEV')
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

YA_API_KEY = os.environ.get('YA_API_KEY')
YA_FOLDER_ID = os.environ.get('YA_FOLDER_ID')
YA_AUTH_TOKEN = os.environ.get('YA_AUTH_TOKEN')

GEMINI_API_KEY=os.environ.get('GEMINI_API_KEY')

CHECK_RIGHTS=os.environ.get('CHECK_RIGHTS', default='False')

USERS_SHEET_ID=os.environ.get('USERS_SHEET_ID')
FEEDBACK_SHEET_ID=os.environ.get('FEEDBACK_SHEET_ID')
GOOGLE_SHEETS_CRED=os.environ.get('GOOGLE_SHEETS_CRED')


JINA_API_KEY=os.environ.get('JINA_API_KEY')


WHISPER_MODEL = os.environ.get('WHISPER_MODEL', default='small')
WHISPER_MODEL_PATH=os.environ.get('WHISPER_MODEL_PATH')

UPD_TIMEOUT = os.environ.get('UPD_TIMEOUT') or 300

def reload_admin_config():
    global CHECK_RIGHTS
    env_vars = dotenv_values(os.path.join(documents_path, 'gv.env'))
    if 'CHECK_RIGHTS' in env_vars:
        CHECK_RIGHTS=env_vars['CHECK_RIGHTS']
    else:
        CHECK_RIGHTS='False'

NOTION_INDEX_FOLDER = os.environ.get('NOTION_INDEX_FOLDER') or "./data/notion_idx"
CHATS_INDEX_FOLDER = os.environ.get('CHATS_INDEX_FOLDER') or "./data/chats_idx"
ASSISTANT_INDEX_FOLDER = os.environ.get('ASSISTANT_INDEX_FOLDER') or "./data/bfts_index"
RERANKING_MODEL = os.environ.get('RERANKING_MODEL') or '/models/bge-reranker-large'
NO_CUDA = os.environ.get('NO_CUDA', "False")

DEBUG_WORKFLOW = (os.environ.get('DEBUG_WORKFLOW', default='False').lower() == 'true')

RETRIEVER_TYPE=os.environ.get('RETRIEVER_TYPE', "teamly")
TEAM_GPT_MODEL=os.environ.get('TEAM_GPT_MODEL', "nano")

LLM_PROVIDER="openai"
BOT_MODE=os.environ.get('BOT_MODE', "polling")
WEBHOOK_BASE=os.environ.get('WEBHOOK_BASE')
WEBAPP_HOST=os.environ.get('WEBAPP_HOST')
WEBAPP_PORT=os.environ.get('WEBAPP_PORT')
WEBHOOK_PATH=os.environ.get('WEBHOOK_PATH')
WEBHOOK_SECRET=os.environ.get('WEBHOOK_SECRET')

MIN_RERANKER_RATIO=os.environ.get('MIN_RERANKER_RATIO', "0.25")