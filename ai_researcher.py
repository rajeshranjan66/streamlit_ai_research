import re
import streamlit as st
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from typing_extensions import TypedDict
import os
from uuid import uuid4

summary_template = """
Summarize the following content into a concise paragraph that directly addresses the query. Ensure the summary 
highlights the key points relevant to the query while maintaining clarity and completeness.
Query: {query}
Content: {content}
"""

generate_response_template = """    
Given the following user query and content, generate a response that directly answers the query using relevant 
information from the content. Ensure that the response is clear, concise, and well-structured. 
Additionally, provide a brief summary of the key points from the response. 
Question: {question} 
Context: {context} 
Answer:
"""
langchain_api_key = st.secrets.get("LANGCHAIN_API_KEY")
unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"AI Research Agent - {unique_id}"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key  # Update to your API ke
print("LangSmith Project id"+unique_id)

class ResearchState(TypedDict):
    query: str
    sources: list[str]
    web_results: list[str]
    summarized_results: list[str]
    response: str

class ResearchStateInput(TypedDict):
    query: str

class ResearchStateOutput(TypedDict):
    sources: list[str]
    response: str

def search_web(state: ResearchState):
    search = TavilySearchResults(max_results=8)
    search_results = search.invoke(state["query"])

    return  {
        "sources": [result['url'] for result in search_results],
        "web_results": [result['content'] for result in search_results]
    }

def summarize_results(state: ResearchState):
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
        # For summarization, we'll process immediately as it's a preprocessing step
        summary = chain.invoke({"query": state["query"], "content": content})
        clean_content = clean_text(summary.content)
        summarized_results.append(clean_content)

    return {"summarized_results": summarized_results}

def generate_response(state: ResearchState):
    model = ChatOpenAI(
        base_url="https://api.deepseek.com/v1",
        api_key=st.secrets["DEEPSEEK_API_KEY"],
        model="deepseek-chat",
        streaming=True
    )
    prompt = ChatPromptTemplate.from_template(generate_response_template)
    chain = prompt | model

    content = "\n\n".join(state["summarized_results"])

    # Return the stream directly
    return {"response": chain.stream({"question": state["query"], "context": content})}

def clean_text(text: str):
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned_text  # Preserves original spacing

builder = StateGraph(
    ResearchState,
    input=ResearchStateInput,
    output=ResearchStateOutput
)

builder.add_node("search_web", search_web)
builder.add_node("summarize_results", summarize_results)
builder.add_node("generate_response", generate_response)

builder.add_edge(START, "search_web")
builder.add_edge("search_web", "summarize_results")
builder.add_edge("summarize_results", "generate_response")
builder.add_edge("generate_response", END)

graph = builder.compile()

st.set_page_config(page_title="AI Research Agent")
#st.title("AI Researcher")
st.subheader("Welcome to AI Research Agent !")
query = st.text_area("Enter your query here:")

if query:
    # Execute the graph normally
    response_state = graph.invoke({"query": query})

    # Create a container for the streaming response
    response_container = st.empty()
    full_response = []

    # Stream the final response only
    for chunk in response_state["response"]:
        chunk_text = clean_text(chunk.content)
        full_response.append(chunk_text)
        response_container.markdown("".join(full_response))

    # Show sources after streaming completes
    st.subheader("Sources:")
    for source in response_state["sources"]:
        st.write(source)