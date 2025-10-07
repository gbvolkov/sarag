import gradio as gr
from gradio import ChatMessage

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from rag import agent
thread_id = 1

def ask_bot(question, history):
    #payload = []
    #for msg in history:
    #    if msg['role'] == "user":
    #        payload.append(HumanMessage(content=msg['content']))
    #    elif msg['role'] == "assistant":
    #        payload.append(AIMessage(content=msg['content']))
    #
    #payload.append(HumanMessage(content=question))
    payload = {"messages": [HumanMessage(content=[{"type": "text", "text": question}])]}
    result = agent.invoke(payload)
    if "messages" in result:
        messages = result["messages"]
        if "content" in messages[-1]:
            return messages[-1].content
    return ""
from typing import List
import asyncio

async def collect_final_text_from_stream(assistant, payload_msg, cfg) -> str:
    """
    Offloads synchronous assistant.stream(...) to a thread
    and returns the concatenated final assistant text.
    """
    def _run_sync():
        final_parts: List[str] = []
        printed_ids = set()
        events = assistant.stream({"messages": [payload_msg]}, cfg, stream_mode="values")
        for event in events:
            msg = event.get("messages") if isinstance(event, dict) else None
            if not msg:
                continue
            if isinstance(msg, list):
                msg = msg[-1]
            mid = getattr(msg, "id", None)
            if mid in printed_ids:
                continue
            if getattr(msg, "type", "") == "ai":
                text = (getattr(msg, "content", "") or "").strip()
                if text:
                    final_parts.append(text)
                    printed_ids.add(mid)
        return "\n".join(final_parts).strip() or "—"

    return await asyncio.to_thread(_run_sync)


def run_agent(query, messages):
    #agent.clear_cache()
    payload_msg = HumanMessage(content=[{"type": "text", "text": query}])
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"messages": [payload_msg]}, config=config)
    msgs = result["messages"]
    final = next((m for m in reversed(msgs) if isinstance(m, AIMessage)), msgs[-1])
    #output = await collect_final_text_from_stream(agent, payload_msg, config)
    
    messages.append(ChatMessage(role="assistant", content=final.content))
    #yield messages
    return messages

def clear_agent_memory():
    global thread_id
    thread_id = thread_id+1

with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# Помощник системного аналитика")
    chatbot = gr.Chatbot(
        type="messages",
        label="Agent",
        avatar_images=(
            None,
            "https://em-content.zobj.net/source/twitter/141/parrot_1f99c.png",
        ),
    )
    query = gr.Textbox(lines=1, label="Chat Message")
    query.submit(run_agent, [query, chatbot], [chatbot])
    chatbot.clear(clear_agent_memory)

if __name__ == "__main__":
    import os
    pid = os.getpid()
    with open(".process", "w") as f:
        f.write(f"{pid}")
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860
    )
