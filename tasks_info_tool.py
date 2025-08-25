from typing_extensions import TypedDict, Annotated, Dict, List

import config

from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import tool

llm_query_gen = ChatOpenAI(model="gpt-4.1", temperature=0)
llm_answer_gen = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

db_tasks = SQLDatabase.from_uri("sqlite:///data/tasks.db")
db_tasks.name = "tasks"

system_message = """
Given an input question, create a syntactically correct {dialect} query to
run to help find the answer. You can order the results by a relevant column to
return the most interesting examples in the database.

Always include into response maimum columns relevant to the question givem. 

Pay attention to use only the column names that you can see in the schema
description. Be careful to not query for columns that do not exist. Also,
pay attention to which column is in which table.

Only use the following tables:
{table_info}
"""

user_prompt = "Question: {input}"

query_prompt_template = ChatPromptTemplate(
    [("system", system_message), ("user", user_prompt)]
)

class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]

def write_query(question: str):
    """Generate SQL query to fetch information."""

    prompt = query_prompt_template.invoke(
        {
            "dialect": db_tasks.dialect,
            "table_info": db_tasks.get_table_info(),
            "input": question
        }
    )
    structured_llm = llm_query_gen.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return result["query"]


def fix_query(query: str, error: str):
    """
    The previous SQL failed.  Regenerate a new query
    taking the DB error into account.
    """
    prompt = (
        f"The following SQL produced an error:\n\n{query}\n\n"
        f"Database error:\n{error}\n\n"
        "Rewrite *only* the SQL so it will execute successfully, following "
        "the same column-name and WHERE-clause rules you already know."
    )

    structured_llm = llm_query_gen.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return result["query"]

def execute_query(query: str):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=db_tasks)

    return execute_query_tool.invoke(query)

def generate_answer(question: str, query: str, result: str):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, provide relevant information from database.\n"
        "If result is empty inform user that there are no records meeting given criteria.\n"
        "Respond with list of flats satisfying criteria\n"
        "Include into response all fields, except technical\n"
        "Include into response price_value, rooms, area_total, renovation and add other fields requested by user\n"
        "Do not include into response any technical fields (for example:ID).\n\n"
        f'Question: {question}\n'
        f'SQL Query: {query }\n'
        f'SQL Result: {result}'
    )
    result = llm_answer_gen.invoke(prompt)
    return result.content

MAX_ATTEMPTS = 3

@tool
def tasks_info(question: str) -> str:
    """Rerurns information about tasks from tasks database. Shall be always used when user asks question about tasks duration, assignment, status and so on.
    Args:
        question: a question user qhants to get answered
    Returns:
        Context from database answering user questions.
        Available fields are:
            "Номер задачи в Битрикс": "bitrix_task_id",
            "Номер ЗНИ": "request_id",
            "Текущий приоритет обращения": "current_ticket_priority",
            "Группа приоритетов": "priority_group",
            "Наименование": "title",
            "Проект Битрикс": "bitrix_project",
            "Инициатор": "initiator",
            "Куратор": "curator",
            "Текущий исполнитель": "current_assignee",
            "ЗНИ текущий этап": "request_current_stage",
            "Статус ЗНИ": "request_status",
            "Дата заведения в системе": "created_date",
            "Плановая дата окончания бизнес-анализа": "planned_business_analysis_end_date",
            "Плановая дата окончания анализа": "planned_analysis_end_date",
            "Плановая дата выполнения": "planned_completion_date",
            "Дата завершения": "completed_date",
            "Код завершения ЗНИ": "request_completion_code",
            "SLA — Дата начала SLA": "sla_start_date",
            "SLA — SLA (кал. дней)": "sla_calendar_days",
            "SLA — Норматив SLA (кал. дней)": "sla_calendar_days_target",
            "SLA — SLA (раб. дней)": "sla_work_days",
            "SLA — Норматив SLA (раб. дней)": "sla_work_days_target",
            "Дата оценки": "estimate_date",
            "Общая оценка, час": "estimate_total_hours",
            "Уточненная оценка, час": "estimate_refined_hours",
            "Факт, час": "actual_hours"
    """
    query = write_query(question)
    attempts = 0
    while True:
        try:
            return execute_query(query)
        except Exception as e:
            if attempts >= MAX_ATTEMPTS:
                raise e
            query = fix_query(query, str(e))
            attempts = attempts + 1

if __name__ == "__main__":
    from pprint import pprint
    question = "Найди ЗНИ, на которое затрачено более 200 часов и которое еще не закрыто."
    query = write_query(question)
    attempts = 0
    while True:
        try:
            result = execute_query(query)
            break
        except Exception as e:
            if attempts >= MAX_ATTEMPTS:
                raise e
            query = fix_query(query, str(e))
            attempts = attempts + 1
    #pprint(result, indent=2)
    print(generate_answer(question, query, result))

    
