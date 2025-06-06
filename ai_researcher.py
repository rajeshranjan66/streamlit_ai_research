import re
import streamlit as st
import time
import logging
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
# from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
import os
from uuid import uuid4
import random
from langgraph.checkpoint.memory import MemorySaver

# Set up logging (for diagnostics)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ExtendedMemorySaver(MemorySaver):
    """
    Extend the MemorySaver to add a checkpoint method.
    This implementation simply keeps a reference to the latest state and logs changes.
    """
    def checkpoint(self, state):
        self.latest_state = state
        logging.info("Checkpoint updated: %s", state)

# Create a singleton instance for this run.
memory_saver = ExtendedMemorySaver()

# A helper to update the state from config.
def update_state_with_config(state: dict, config: dict) -> dict:
    """
    Merge configuration-provided updates into the state.
    For example, if config contains a key "state_update", its items will be merged into state.
    """
    if config is None:
        return state
    state_update = config.get("state_update", {})
    if state_update:
        logging.info("***********************Updating state with config: %s", state_update)
        state.update(state_update)
    return state

# Initialize Streamlit session state for chat history and streaming flag.
if "messages" not in st.session_state:
    st.session_state.messages = []  # List[dict] with entries like {"role": "...", "content": "..."}
if "stop_streaming" not in st.session_state:
    st.session_state.stop_streaming = False

# Define some funny error messages
funny_messages = [
    "Oops! Something went wrong... Maybe it's a feature, not a bug? 🤖",
    "Error 404: AI's confidence not found. Let's try again! 😵‍💫",
    "Houston, we have a problem! But don't worry, it's just a minor hiccup. 🚀",
    "Well, this is awkward... Let's pretend this never happened. 😅",
    "Looks like we summoned the error gods today. Let's appease them with a retry! 🔄"
]

# Updated prompt templates remain unchanged.
summary_template = """
Summarize the following content into a well-structured and concise paragraph that directly addresses the user's query. 
Ensure the summary highlights the most **relevant and key points**, presenting them in a **clear, coherent, and informative** manner. 
Avoid unnecessary details while maintaining completeness, factual accuracy, and readability.

Focus on:
- Extracting **critical insights** related to the query.
- Providing a **concise yet meaningful** synthesis of the content.
- Maintaining a **natural and engaging tone** for better readability.

Query: {query}
Content: {content}
"""

generate_response_template = """    
Generate a **clear, concise, and well-structured** response based on the given user query and contextual content. 
Ensure that the answer directly addresses the query using the **most relevant and accurate information** extracted from the content. 

Key Guidelines:
- **Prioritize relevance**—focus on the most **useful and insightful** details.
- **Maintain readability**—use a **natural, engaging tone** while ensuring clarity.
- **Structure effectively**—organize the response logically for easy comprehension.
- **Summarize key points**—provide a brief, well-defined **summary** at the end to highlight essential takeaways.

Question: {question}  
Context: {context}  
Answer:
"""

# Configuration and setup code
langchain_api_key = st.secrets.get("LANGCHAIN_API_KEY")
unique_id = uuid4().hex[0:8]
os.environ.update({
    "LANGCHAIN_TRACING_V2": "true",
    "LANGCHAIN_PROJECT": f"AI Research Agent - {unique_id}",
    "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
    "LANGCHAIN_API_KEY": langchain_api_key
})

# Update your ResearchState to include conversation history.
class ResearchState(TypedDict):
    query: str
    sources: list[str]
    web_results: list[str]
    summarized_results: list[str]
    response: str
    history: list[dict]  # Each history dict: {"role": "user"/"assistant", "content": "..."}
    thread_id: str  # Unique identifier for the conversation thread

# Node definitions now accept and retain the conversation history.

def search_web(state: ResearchState, config: dict = None):
    # Merge any updates from config.
    print("S^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^tarting web search with state:", state)
    state = update_state_with_config(state, config)
    if st.session_state.stop_streaming:
        raise StreamingStoppedError()

    logging.info("Starting web search for query: %s", state["query"])
    search = TavilySearchResults(max_results=3)
    search_results = search.invoke(state["query"])

    # Update the state with search results.
    state["sources"] = [result['url'] for result in search_results]
    state["web_results"] = [result['content'] for result in search_results]
    logging.info("Web search completed. Sources: %s", state["sources"])

    # Checkpoint the state (including conversation history).
    memory_saver.checkpoint(state)
    return state

def summarize_results(state: ResearchState, config: dict = None):
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^Starting summarization with state:", state)
    state = update_state_with_config(state, config)
    if st.session_state.stop_streaming:
        raise StreamingStoppedError()

    logging.info("Starting summarization for query: %s", state["query"])
    model = ChatOpenAI(
        base_url="https://api.deepseek.com/v1",
        api_key=st.secrets["DEEPSEEK_API_KEY"],
        model="deepseek-chat",
        streaming=True
    )
    prompt = ChatPromptTemplate.from_template(summary_template)
    chain = prompt | model
    summarized_results = []
    for content in state["web_results"]:
        summary = chain.invoke({"query": state["query"], "content": content})
        clean_summary = clean_text(summary.content)
        summarized_results.append(clean_summary)
        logging.info("Summarized content: %s", clean_summary)

    state["summarized_results"] = summarized_results
    memory_saver.checkpoint(state)
    return state

def generate_response(state: ResearchState, config: dict = None):
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^Starting response generation with state:", state)
    state = update_state_with_config(state, config)
    if st.session_state.stop_streaming:
        raise StreamingStoppedError()

    logging.info("Generating response for query: %s", state["query"])
    model = ChatOpenAI(
        base_url="https://api.deepseek.com/v1",
        api_key=st.secrets["DEEPSEEK_API_KEY"],
        model="deepseek-chat",
        streaming=True
    )
    prompt = ChatPromptTemplate.from_template(generate_response_template)
    chain = prompt | model
    content = "\n\n".join(clean_text(result) for result in state["summarized_results"])
    memory_saver.checkpoint(state)
    # The response is returned as a streaming generator.
    return {"response": chain.stream({"question": state["query"], "context": content})}

def clean_text(text: str):
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned_text

# Build the LangGraph workflow.
builder = StateGraph(ResearchState)
builder.add_node("search_web", search_web)
builder.add_node("summarize_results", summarize_results)
builder.add_node("generate_response", generate_response)
builder.set_entry_point("search_web")
builder.add_edge("search_web", "summarize_results")
builder.add_edge("summarize_results", "generate_response")
builder.set_finish_point("generate_response")
graph = builder.compile()

# -------------------------------------------------------------------
# Streamlit UI code (updated to set initial state with conversation history)
# -------------------------------------------------------------------
st.set_page_config(page_title="AI Research Agent", layout="wide")
st.subheader("Welcome to AI Research Agent")

with st.sidebar:
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
    if st.button("🛑 Stop Streaming"):
        st.session_state.stop_streaming = True

# Display chat history from the session.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Chat input and processing.
if prompt := st.chat_input("Enter your research query..."):
    st.session_state.stop_streaming = False  # Reset the streaming flag for new queries.

    # Add the new user message to the session state.
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# Prepare our custom error class.
class StreamingStoppedError(Exception):
    """Custom exception raised when streaming is stopped."""
    def __init__(self, message="Streaming has been stopped"):
        super().__init__(message)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = uuid4().hex

# When processing a new chat message:
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    try:
        user_prompt = st.session_state.messages[-1]["content"]

        # Retrieve the previous state from the custom memory (if available)
        # and if the thread_id matches, then use that as the base.
        if "thread_id" in st.session_state and hasattr(memory_saver, "latest_state"):
            stored_state = memory_saver.latest_state
            # Ensure that the stored state belongs to the same thread.
            if stored_state.get("thread_id") == st.session_state.thread_id:
                initial_state = stored_state
            else:
                initial_state = {}
        else:
            initial_state = {}

        # Update the initial state with the new query and conversation history.
        # Here, we merge the persistent state with what’s currently displayed in session.
        initial_state.update({
            "query": user_prompt,
            "sources": initial_state.get("sources", []),
            "web_results": initial_state.get("web_results", []),
            "summarized_results": initial_state.get("summarized_results", []),
            "response": initial_state.get("response", ""),
            "history": st.session_state.messages.copy(),  # complete conversation history
            "thread_id": st.session_state.thread_id
        })

        # Prepare any additional config data if needed.
        config = {
            "state_update": {"custom_note": "Added via config"},
            "configurable": {"thread_id": st.session_state.thread_id}  # redundant if in state, but available if needed.
        }

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            sources_placeholder = st.empty()

        with st.spinner("Researching..."):
            start_time = time.time()
            # Pass the merged state and config to the graph.
            response_state = graph.invoke(initial_state, config=config)
            full_response = []

            for chunk in response_state["response"]:
                if st.session_state.stop_streaming:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "".join(full_response)
                    })
                    raise StreamingStoppedError()
                chunk_text = clean_text(chunk.content)
                full_response.append(chunk_text)

                # Update the assistant’s message in session state.
                if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
                    st.session_state.messages[-1]["content"] = "".join(full_response)
                else:
                    st.session_state.messages.append({"role": "assistant", "content": "".join(full_response)})

                response_placeholder.markdown("".join(full_response), unsafe_allow_html=True)

            final_text = "".join(full_response)
            duration = time.time() - start_time
            footnote = f"<div style='font-size:0.8em; color:#666; margin-top:10px;'>⏱️ Processed in {duration:.2f}s</div>"
            formatted_response = f"{final_text}\n{footnote}"
            response_placeholder.markdown(formatted_response, unsafe_allow_html=True)
            sources_placeholder.markdown(
                "🔗 **Sources**:\n" + "\n".join(f"- {src}" for src in response_state["sources"])
            )

            # Save the final assistant response and update the history.
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"{final_text}\n{footnote}\n\n🔗 **Sources**:\n" +
                           "\n".join(f"- {source}" for source in response_state["sources"])
            })
            # Also update the state history.
            response_state["history"] = st.session_state.messages.copy()

        st.session_state.stop_streaming = False
    except StreamingStoppedError as e:
        st.error(str(e))
        st.session_state.stop_streaming = False
    except Exception as exc:
        logging.error("Exception occurred during processing: %s", exc)
        st.error(random.choice(funny_messages))
