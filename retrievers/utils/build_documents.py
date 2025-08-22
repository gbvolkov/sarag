import pandas as pd
from langchain_core.documents import Document  # LangChain Document class

def get_documents_for_sd_qa(df: pd.DataFrame) -> list[Document]:
    # Create a list of Documents from each row
    documents = []
    for _, row in df.iterrows():
        # Combine relevant fields into the page_content for retrieval
        content = (
            f"It System: {row['it_system']}\n"
            f"Problem Description: {row['problem_description']}\n"
            f"Problem Solution: {row['problem_solution']}\n\n"
            f"Ссылка на статью:https://kb.ileasing.ru/space/{row['space_id']}/article/{row['article_id']}"
        )
        doc = Document(
            page_content=content,
            metadata={
                "it_system": row["it_system"],
                "problem_description": row["problem_description"],
                "problem_solution": row["problem_solution"],
                "space_id": row["space_id"],
                "article_id": row["article_id"],
                "article_title": row["article_title"],
                "source": "sd_qa_table"
            }
        )
        documents.append(doc)
    return documents

def get_documents_for_sd_tickets(df: pd.DataFrame) -> list[Document]:
    # Create a list of Documents from each row
    documents = []
    for _, row in df.iterrows():
        # Combine relevant fields into the page_content for retrieval
        content = (
            f"Ticket number: {row['ticket_no']}\n"
            f"Ticket topic: {row['topic']}\n"
            f"Problem: {row['problem']}\n\n"
            f"Solution: {row['solution']}\n\n"
            f"Ссылка на статью:https://kb.ileasing.ru/space/{row['space_id']}/article/{row['article_id']}"
        )
        doc = Document(
            page_content=content,
            metadata={
                "ticket_no": row["ticket_no"],
                "topic": row["topic"],
                "ticket_dt": row["ticket_dt"],
                "ticket_type": row["ticket_type"],
                "problem": row["problem"],
                "solution": row["solution"],
                "space_id": row["space_id"],
                "article_id": row["article_id"],
                "article_title": row["article_title"],
                "source": "sd_tickets_table"
            }
        )
        documents.append(doc)
    return documents

def get_documents_for_glossary(df: pd.DataFrame) -> list[Document]:
    # Create a list of Documents from each row
    documents = []
    for _, row in df.iterrows():
        # Combine relevant fields into the page_content for retrieval
        content = (
            f"Term: {row['term']}\n"
            f"Definition: {row['definition']}\n"
            f"Ссылка на статью:https://kb.ileasing.ru/space/{row['space_id']}/article/{row['article_id']}"
        )
        doc = Document(
            page_content=content,
            metadata={
                "term": row["term"],
                "definition": row["definition"],
                "space_id": row["space_id"],
                "article_id": row["article_id"],
                "article_title": row["article_title"],
                "source": row["section"]
            }
        )
        documents.append(doc)
    return documents