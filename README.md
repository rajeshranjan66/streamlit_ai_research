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

- Python 3.8+
- [Streamlit](https://streamlit.io/)
- [LangChain](https://github.com/langchain-ai/langchain)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- Tavily, DeepSeek, and (optionally) OpenAI API keys

### Installation

bash
git clone https://github.com/rajeshranjan66/streamlit_ai_research.git
cd streamlit_ai_research
pip install -r requirements.txt

--



### 🛠️ Setup API Keys
TOML
# .streamlit/secrets.toml

LANGCHAIN_API_KEY = "your_langchain_api_key"
DEEPSEEK_API_KEY = "your_deepseek_api_key"
Place this file at .streamlit/secrets.toml in your project directory.
Replace the placeholder values with your actual API keys.

--

💬 Usage
Enter your research query in the chat input at the bottom of the app.
Clear Chat History: Use the sidebar button to reset the conversation.
Stop Streaming: Use the sidebar button to stop a long or unwanted AI response in real time.
Sources are shown with each answer for transparency.
