import logging

from typing import List, Any, Optional, Dict, Tuple, TypedDict, Annotated
import os
import pickle
import torch
from langchain_community.document_loaders import NotionDBLoader
from langchain_community.document_loaders import NotionDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.docstore.document import Document
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.storage import InMemoryByteStore
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain.retrievers import MergerRetriever
from retrievers.teamly_retriever import (
    TeamlyRetriever,
    TeamlyRetriever_Tickets,
    TeamlyRetriever_Glossary,
    TeamlyContextualCompressionRetriever
)
from retrievers.head_documents_loader import _build_heads_by_source, _dedupe_docs
import config


from copy import deepcopy

class CrossEncoderRerankerWithScores(CrossEncoderReranker):
    min_ratio: int = 0

    def __init__(self, *args, min_ratio: float = 0.00, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_ratio=min_ratio
    def compress_documents(self, documents, query, callbacks=None):
        # compute scores
        scores = self.model.score([(query, d.page_content) for d in documents])
        # attach to metadata (without mutating originals)
        docs = []
        for d, s in zip(documents, scores):
            d2 = deepcopy(d)
            d2.metadata = {**(d2.metadata or {}), "rerank_score": float(s)}
            docs.append(d2)
        max_s = max(scores)
        threshold = self.min_ratio*max_s
        passed_docs = [d for d in docs if d.metadata["rerank_score"] >= threshold]
        # sort by score desc and keep top_n
        passed_docs.sort(key=lambda d: d.metadata["rerank_score"], reverse=True)
        return passed_docs[: self.top_n]

# Global instances and refreshable Teamly Retriever for hot index updates
_teamly_retriever_instance: Optional[TeamlyRetriever] = None
_teamly_retriever_tickets_instance : Optional[TeamlyRetriever_Tickets] = None
_teamly_retriever_glossary_instance : Optional[TeamlyRetriever_Glossary] = None
#_teamly_compression_retriever_instance: Optional[TeamlyContextualCompressionRetriever] = None

def load_vectorstore(file_path: str, embedding_model_name: str) -> FAISS:
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No vectorstore found at {file_path}")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={"device": device})
    return FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)


def get_retriever_object_faiss_chunked():
    MAX_RETRIEVALS = 3
    k = 5
    bft_store_path = config.ASSISTANT_INDEX_FOLDER
    itil_store_path = config.ITIL_INDEX_FOLDER
    bft_vs = load_vectorstore(bft_store_path, config.EMBEDDING_MODEL)
    
    itil_vs = load_vectorstore(itil_store_path, config.EMBEDDING_MODEL)
    
    with open(f'{itil_store_path}/docstore.pkl', 'rb') as file:
        documents = pickle.load(file)
    doc_ids = [doc.metadata.get('relative_path', '') for doc in documents]
    store = InMemoryByteStore()
    id_key = "relative_path"
    itil_retriever = MultiVectorRetriever(
        vectorstore=itil_vs,
        byte_store=store,
        id_key=id_key,
        search_kwargs={"k": 2},
    )
    itil_retriever.docstore.mset(list(zip(doc_ids, documents)))

    #itil_retriever = itil_vs.as_retriever(search_kwargs={"k": k})
    # Load docstore once and pre-index "head" docs by source
    with open(os.path.join(bft_store_path, "docstore.pkl"), "rb") as f:
        docstore = pickle.load(f)
    head_store = _build_heads_by_source(docstore)

    ensemble_retiriever = EnsembleRetriever(
        retrievers=[bft_vs.as_retriever(search_kwargs={"k": k}),
                    itil_retriever],
        weights=[0.6, 0.4]  # adjust to favor text vs. images
    )

    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    reranker_model = HuggingFaceCrossEncoder(
        model_name=config.RERANKING_MODEL,
        model_kwargs={'trust_remote_code': True, "device": device}
    )
    reranker = CrossEncoderRerankerWithScores(model=reranker_model, top_n=MAX_RETRIEVALS, min_ratio=float(config.MIN_RERANKER_RATIO))
    retriever = ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=ensemble_retiriever
    )
    return retriever, head_store

def get_retriever_faiss_chunked():
    retriever, head_store = get_retriever_object_faiss_chunked()
    MAX_RETRIEVALS = 3
    def search(query: str) -> List[Document]:
        # 1) do the regular retrieval once
        retrieved = retriever.invoke(query, search_kwargs={"k": MAX_RETRIEVALS})
        # 2) collect unique sources from retrieved docs
        sources = {
            d.metadata.get("source")
            for d in retrieved
            if d.metadata and d.metadata.get("source")
        }
        if not sources:
            return retrieved
        # 3) fetch head docs from the prebuilt index (no vector search)
        head_docs = [h for s in sources for h in head_store.get(s, [])]
        # 4) merge + dedupe
        return _dedupe_docs(retrieved + head_docs)
    def retrieve_requirements(bitrix_task_id: str) -> List[Document]:
        return [
            doc[0]
            for doc in head_store.values()
            if isinstance(getattr(doc[0], "metadata", None), dict)
            and str(bitrix_task_id) in doc[0].metadata.get("task_no")
        ]
    return search, retrieve_requirements


def get_retriever():
    return get_retriever_faiss_chunked()

# Initialize the search function with the selected retrieverx
search, retrieve_requirements = get_retriever()


def refresh_indexes():
    """Refresh the indexes of the active retriever (e.g., rebuild Teamly FAISS and BM25 indexes)."""
    logging.info("Refreshing faiss indexes...")
    if config.RETRIEVER_TYPE == "teamly" and _teamly_retriever_instance:
        _teamly_retriever_instance.refresh()
    if _teamly_retriever_tickets_instance:
        _teamly_retriever_tickets_instance.refresh()
    logging.info("...complete refreshing faiss indexes.")

def get_search_tool():
    @tool
    def search_kb(query: str) -> str: #, bitrix_task_id: Optional[str]) -> str:
        """Retrieves from knowledgebase context suitable for the query. Shall be always used when user asks question.
        Args:
            query: a query to knowledgebase which helps answer user's question. 
                Include into query task name, user's question, Number of the task in bitrix, request id and other available inormation
        Returns:
            Context from knowledgebase suitable for the query.
        """
        found_docs = search(query)
        if found_docs:
            result = "\n\n".join([doc.page_content for doc in found_docs[:30]])
            return result
        else:
            return "No matching information found."
    return search_kb

def get_retrieve_requirements_tool():
    @tool
    def retrieve_task_requirements(bitrix_task_id: str) -> str:
        """Retrieves functional requirements related to a specific bitrix task. Shall be always used when user asks question for a specific task.
        Args:
            bitrix_task_id: Number of the task in Bitrix)
        Returns:
            Context from functional requirements storage related to a specific bitrix task.
        """
        found_docs = retrieve_requirements(bitrix_task_id)
        if found_docs:
            result = "\n\n".join([doc.page_content for doc in found_docs[:30]])
            return result
        else:
            return "No matching information found."
    return retrieve_task_requirements

def get_search_tool():
    @tool
    def search_kb(query: str) -> str: #, bitrix_task_id: Optional[str]) -> str:
        """Retrieves from knowledgebase context suitable for the query. Shall be always used when user asks question.
        Args:
            query: a query to knowledgebase which helps answer user's question. 
                Expand user's query with maximum of available information (task name, Number of the task in bitrix, request id and so on)
        Returns:
            Context from knowledgebase suitable for the query.
        """
        found_docs = search(query)
        if found_docs:
            result = "\n\n".join([doc.page_content for doc in found_docs[:30]])
            return result
        else:
            return "No matching information found."
    return search_kb


def get_term_and_defition_tools():
    MAX_RETRIEVALS = 3
    global _teamly_retriever_glossary_instance

    _teamly_retriever_glossary_instance = TeamlyRetriever_Glossary("./auth_glossary.json", k=MAX_RETRIEVALS)
    
    @tool
    def lookup_term(term: str) -> str:
        """
        Look up the definition of a term or abbreviation in the reference dictionary.

        This tool is designed to retrieve the meaning of either a full term 
        or an abbreviation from a predefined reference source. 
        All abbreviations in the reference are stored in uppercase. 
        All terms in the reference are stored in singular nominative case. 

        The input must strictly follow these conventions:
        - Abbreviations: uppercase only (e.g., "HTTP", "NASA", "АД").
        - Terms: singular nominative case (e.g., "server", "network", "лизинговая заявка").

        Args:
            name (str): The term or abbreviation to look up.
                Must match the format and casing conventions of the reference.

        Returns:
            str: The definition or description of the provided term or abbreviation.
                Currently returns a constant placeholder string.
        """
        found_docs = _teamly_retriever_glossary_instance.invoke(term)
        if found_docs:
            result = "\n\n".join([doc.page_content for doc in found_docs[:30]])
            return result
        else:
            return "No matching information found."
    return lookup_term

if __name__ == '__main__':
    search_kb = get_search_tool()
    answer = search_kb("Кто такие кей юзеры?")
    print(answer)