import re

import streamlit as st
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph
from typing_extensions import TypedDict

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
    search = TavilySearchResults(max_results=3)
    search_results = search.invoke(state["query"])

    return  {
        "sources": [result['url'] for result in search_results],
        "web_results": [result['content'] for result in search_results]
    }

def summarize_results(state: ResearchState):
    model = ChatOllama(model="deepseek-r1:1.5b")
    prompt = ChatPromptTemplate.from_template(summary_template)
    chain = prompt | model

    summarized_results = []
    for content in state["web_results"]:
        summary = chain.invoke({"query": state["query"], "content": content})
        clean_content = clean_text(summary.content)
        summarized_results.append(clean_content)

    return {
        "summarized_results": summarized_results
    }

def generate_response(state: ResearchState):
    model = ChatOllama(model="deepseek-r1:1.5b")
    prompt = ChatPromptTemplate.from_template(generate_response_template)
    chain = prompt | model

    content = "\n\n".join([summary for summary in state["summarized_results"]])

    return {
        "response": chain.invoke({"question": state["query"], "context": content})
    }

def clean_text(text: str):
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned_text.strip()

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

st.title("AI Researcher")
query = st.text_input("Enter your research query:")

if query:
    response_state = graph.invoke({"query": query})
    st.write(clean_text(response_state["response"].content))

    st.subheader("Sources:")
    for source in response_state["sources"]:
        st.write(source)
