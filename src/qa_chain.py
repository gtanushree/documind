from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableBranch, RunnablePassthrough
from langchain_groq import ChatGroq

from .config import settings
from .vector_store import get_vector_store

CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the conversation so far and a follow-up question, rephrase the "
            "follow-up question into a standalone question that contains all "
            "necessary context. Do not answer it — only rephrase. If it is "
            "already standalone, return it unchanged.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are DocuMind, a precise document Q&A assistant. Answer the "
            "question using ONLY the context below, drawn from the user's "
            "uploaded PDFs. If the answer isn't contained in the context, say "
            "you couldn't find it in the documents rather than guessing. Cite "
            "which document(s) you drew on when it's useful.\n\n"
            "Context:\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{d.metadata.get('source', 'unknown')}, page {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )


def build_chain(namespace: str, k: int | None = None) -> Runnable:
    k = k or settings.retrieval_k
    retriever = get_vector_store(namespace).as_retriever(search_kwargs={"k": k})
    llm = ChatGroq(model=settings.llm_model, groq_api_key=settings.groq_api_key, temperature=0.1)

    condense_chain = CONDENSE_PROMPT | llm | StrOutputParser()

    history_aware_retriever = RunnableBranch(
        (lambda x: not x.get("chat_history"), (lambda x: x["input"]) | retriever),
        condense_chain | retriever,
    )

    document_chain = (
        RunnablePassthrough.assign(context=lambda x: _format_docs(x["context"]))
        | ANSWER_PROMPT
        | llm
        | StrOutputParser()
    )

    return RunnablePassthrough.assign(context=history_aware_retriever).assign(answer=document_chain)


def to_lc_history(chat_history: list[dict]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for turn in chat_history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    return messages


def ask(chain: Runnable, question: str, chat_history: list[dict]) -> dict:
    result = chain.invoke({"input": question, "chat_history": to_lc_history(chat_history)})

    sources = []
    seen = set()
    for doc in result.get("context", []):
        key = (doc.metadata.get("source"), doc.metadata.get("page"))
        if key in seen:
            continue
        seen.add(key)
        snippet = doc.page_content[:280].strip()
        if len(doc.page_content) > 280:
            snippet += "…"
        sources.append({"source": doc.metadata.get("source", "unknown"), "page": doc.metadata.get("page", "?"), "snippet": snippet})

    return {"answer": result["answer"], "sources": sources}
