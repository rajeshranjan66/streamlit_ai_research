# AI Research Agent

**AI Research Agent** is a Streamlit-powered web application that leverages advanced AI and web search tools to help users research any topic efficiently. By combining LangChain, DeepSeek, and Tavily Search, the app fetches, summarizes, and synthesizes web content into concise, insightful answers in a conversational interface.

---

## 🚀 Features

- Conversational, chat-style research Q&A
- Web search integration (Tavily Search)
- AI-powered summarization (DeepSeek, OpenAI)
- Streaming responses for live updates
- Chat history memory
- Sidebar controls (clear chat, stop streaming)
- Fun error messages

---

## 🏗️ How It Works

1. User submits a research query.
2. Web search gathers relevant results.
3. AI summarizes each result.
4. Final answer is generated and sources are cited.

---

## 🛠️ Getting Started

### Prerequisites

- Python 3.12+
- [Streamlit](https://streamlit.io/)
- [LangChain](https://github.com/langchain-ai/langchain)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- Tavily, DeepSeek, and (optionally) OpenAI API keys

### Installation

```bash
git clone https://github.com/rajeshranjan66/streamlit_ai_research.git
cd streamlit_ai_research
pip install -r requirements.txt
```
### 🔑 Setup API Keys
Create a file at .streamlit/secrets.toml in your project directory and add:

LANGCHAIN_API_KEY = "your_langchain_api_key"
DEEPSEEK_API_KEY = "your_deepseek_api_key"

Replace the placeholder values with your actual API keys.

### 💬 Usage
Type your research question into the chat input at the bottom of the app.
Use the sidebar to clear the chat history or stop streaming a response.
Answers include relevant sources for transparency.

Start the app with:
```bash
streamlit run ai_researcher.py
```

---
## 🧑‍💻 Code Structure and Explanation
This section provides a block-by-block walkthrough of ai_researcher.py, explaining each part for new developers.

### 1. Imports and Session State
Imports Streamlit, LangChain, LangGraph, Tavily, DeepSeek, and utility modules. <br>
Initializes session state variables for chat history (messages) and streaming control (stop_streaming).<br>
Defines fun error messages for a better user experience.<br>

### 2. Prompt Templates
   
summary_template: Used to condense web search results into concise summaries.<br>
generate_response_template: Synthesizes summaries into a single, structured, relevant answer.<br>

### 3. API Key and Environment Configuration
Reads secrets from Streamlit’s secret storage.<br>
Sets environment variables for LangChain tracing and project settings.<br>

### 4. Core Functions
   
search_web: Uses Tavily to perform web search.<br>
summarize_results: Uses DeepSeek to summarize web results.<br>
generate_response: Synthesizes all summaries into one answer and streams the result.<br>
clean_text: Removes unwanted tags or formatting from AI output.<br>

### 5. Workflow with LangGraph
   
Uses LangGraph’s StateGraph to chain workflow steps:<br>
search_web → summarize_results → generate_response<br>
Modular and easy to extend.<br>

### 6. Streamlit UI Handling
    
Sidebar: clear chat, stop streaming.<br>
Chat display: user and AI messages.<br>
Chat input: for research queries.<br>
Message processing: runs workflow, streams responses, shows sources and processing time.<br>
Error handling: fun error messages and user interruption support.<br>

### 7. Error Handling
    
StreamingStoppedError: Custom exception for stopping output mid-response.<br>
Randomized error messages for a friendly experience.<br>

### 8. Tips for Developers
Add new LLMs or APIs by updating summarization or response functions.<br>
Add workflow steps by extending the LangGraph workflow.<br>
Refine prompt templates for different answer styles.<br>
Expand error handling as needed.<br>

### 9.License

MIT License

### 10.Acknowledgments

LangChain<br>
Streamlit<br>
DeepSeek<br>
Tavily<br>


