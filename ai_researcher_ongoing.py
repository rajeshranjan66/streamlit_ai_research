import re
import streamlit as st
import time
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
#from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
import os
from uuid import uuid4
import random


# Initialize session state for chat history and streaming control
if "messages" not in st.session_state:
    st.session_state.messages = []
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
#prompt to be used by different nodes for creating summary and generating final response
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

# Configuration and setup
langchain_api_key = st.secrets.get("LANGCHAIN_API_KEY")
unique_id = uuid4().hex[0:8]
os.environ.update({
    "LANGCHAIN_TRACING_V2": "true",
    "LANGCHAIN_PROJECT": f"AI Research Agent - {unique_id}",
    "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
    "LANGCHAIN_API_KEY": langchain_api_key
})


# Class definition
class ResearchState(TypedDict):
    query: str
    sources: list[str]
    web_results: list[str]
    summarized_results: list[str]
    response: str

class StreamingStoppedError(Exception):
    """Custom exception raised when streaming is stopped."""
    def __init__(self, message="Streaming has been stopped"):
        super().__init__(message)

# Functions
def search_web(state: ResearchState):
    if st.session_state.stop_streaming:
        raise StreamingStoppedError()
    search = TavilySearchResults(max_results=3)
    search_results = search.invoke(state["query"])
    return {"sources": [r['url'] for r in search_results], "web_results": [r['content'] for r in search_results]}

def summarize_results(state: ResearchState):
    if st.session_state.stop_streaming:
        raise StreamingStoppedError()
    model = ChatOpenAI(api_key=st.secrets["DEEPSEEK_API_KEY"], model="deepseek-chat")
    prompt = ChatPromptTemplate.from_template(summary_template)
    chain = prompt | model
    return {"summarized_results": [chain.invoke({"query": state["query"], "content": c}).content for c in state["web_results"]]}

def generate_response(state: ResearchState):
    if st.session_state.stop_streaming:
        raise StreamingStoppedError()
    model = ChatOpenAI(api_key=st.secrets["DEEPSEEK_API_KEY"], model="deepseek-chat", streaming=True)
    prompt = ChatPromptTemplate.from_template(generate_response_template)
    chain = prompt | model
    return {"response": chain.stream({"question": state["query"], "context": "\n".join(state["summarized_results"])})}

# LangGraph workflow
builder = StateGraph(ResearchState)
builder.add_node("search_web", search_web)
builder.add_node("summarize_results", summarize_results)
builder.add_node("generate_response", generate_response)
builder.set_entry_point("search_web")
builder.add_edge("search_web", "summarize_results")
builder.add_edge("summarize_results", "generate_response")
builder.set_finish_point("generate_response")
graph = builder.compile()

# UI Setup
st.subheader("Welcome to AI Research Agent")

# Sidebar
with st.sidebar:
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
    if st.button("🛑 Stop Streaming"):
        st.session_state.stop_streaming = True

# Chat Display
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Chat Input
if prompt := st.chat_input("Enter your research query..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# Process AI response
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    try:
        # Reset stop_streaming at the start of a new query
        if st.session_state.stop_streaming:
            st.session_state.stop_streaming = False

        user_prompt = st.session_state.messages[-1]["content"]

        if st.session_state.stop_streaming:  # Stop execution before research begins
            raise StreamingStoppedError()  # Interrupt research process

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            sources_placeholder = st.empty()

        with st.spinner("Researching..."):
            start_time = time.time()
            response_state = graph.invoke({"query": user_prompt})
            full_response = []

            for chunk in response_state["response"]:
                if st.session_state.stop_streaming:
                    raise StreamingStoppedError()  # Interrupt response generation
                chunk_text = chunk.content
                full_response.append(chunk_text)
                response_placeholder.markdown("".join(full_response), unsafe_allow_html=True)

            final_text = "".join(full_response)
            duration = time.time() - start_time
            footnote = f"<div style='font-size:0.8em; color:#666; margin-top:10px;'>⏱️ Processed in {duration:.2f}s</div>"
            response_placeholder.markdown(final_text + footnote, unsafe_allow_html=True)
            sources_placeholder.markdown(
                "🔗 **Sources**:\n" + "\n".join(f"- {src}" for src in response_state["sources"]))

            st.session_state.messages.append({
                "role": "assistant",
                "content": f"{final_text}\n{footnote}\n\n🔗 **Sources**:\n" +
                           "\n".join(f"- {source}" for source in response_state["sources"])
            })

            st.session_state.stop_streaming = False  # Reset flag after response

    except StreamingStoppedError as e:
        st.error(str(e))  # Display "Streaming has been stopped by the user."
        st.session_state.stop_streaming = False  # Reset flag after stopping

    #except Exception:
    #    st.error(random.choice(funny_messages))  # Show a random funny error message

