import streamlit as st
import os
from dotenv import load_dotenv
from typing import Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from tavily import TavilyClient
from langchain_core.tools import tool

# Page Configuration
st.set_page_config(
    page_title="Skin Cancer AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for skin cancer theme (peach, pink, medical blue)
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #FFF5F0 0%, #FFE8E0 100%);
    }
    
    /* Fix all text colors in main area */
    .main .block-container {
        color: #2C3E50 !important;
    }
    
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {
        color: #2C3E50 !important;
    }
    
    .main p, .main label, .main span, .main div {
        color: #2C3E50 !important;
    }
    
    .stMarkdown {
        color: #2C3E50 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FF9B85 0%, #FFB4A3 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: #2C3E50 !important;
    }
    
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] span, [data-testid="stSidebar"] li {
        color: #2C3E50 !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown strong {
        color: #1A252F !important;
        font-weight: bold;
    }
    
    [data-testid="stSidebar"] button {
        color: #2C3E50 !important;
        font-weight: 600;
    }
    
    /* Chat messages */
    .user-message {
        background: linear-gradient(135deg, #4A90E2 0%, #5BA3F5 100%);
        color: #1A252F !important;
        padding: 15px 20px;
        border-radius: 18px 18px 5px 18px;
        margin: 10px 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 2px 8px rgba(74, 144, 226, 0.3);
        font-weight: 500;
    }
    
    .bot-message {
        background: white;
        color: #2C3E50;
        padding: 15px 20px;
        border-radius: 18px 18px 18px 5px;
        margin: 10px 0;
        max-width: 80%;
        border-left: 4px solid #FF9B85;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        line-height: 1.6;
    }
    
    .bot-message-container {
        background: white;
        padding: 20px;
        border-radius: 18px 18px 18px 5px;
        margin: 10px 0;
        max-width: 80%;
        border-left: 4px solid #FF9B85;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        color: #2C3E50 !important;
    }
    
    .bot-message-container p, .bot-message-container div, 
    .bot-message-container span, .bot-message-container li {
        color: #2C3E50 !important;
    }
    
    .bot-message-container strong {
        color: #FF7A5C !important;
    }
    
    .bot-message-container code {
        background: #F8F9FA !important;
        color: #E83E8C !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
        font-family: 'Courier New', monospace !important;
    }
    
    .bot-message-container pre {
        background: #2D2D2D !important;
        color: #F8F8F2 !important;
        padding: 15px !important;
        border-radius: 8px !important;
        overflow-x: auto !important;
        border-left: 3px solid #FF9B85 !important;
    }
    
    .bot-message-container pre code {
        background: transparent !important;
        color: #F8F8F2 !important;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #FF9B85 0%, #FFB4A3 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(255, 155, 133, 0.3);
    }
    
    .main-header h1 {
        color: #2C3E50 !important;
        font-size: 2.5em;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(255,255,255,0.3);
        font-weight: bold;
    }
    
    .main-header p {
        color: #1A252F !important;
        font-size: 1.2em;
        margin-top: 10px;
        font-weight: 500;
    }
    
    /* Input box styling */
    .stTextInput input {
        border-radius: 25px;
        border: 2px solid #FF9B85;
        padding: 12px 20px;
        font-size: 16px;
        color: #2C3E50 !important;
        background: white !important;
    }
    
    .stTextInput input::placeholder {
        color: #95A5A6 !important;
    }
    
    .stTextInput input:focus {
        border-color: #FF7A5C;
        box-shadow: 0 0 10px rgba(255, 155, 133, 0.3);
    }
    
    .stTextInput label {
        color: #2C3E50 !important;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #FF9B85 0%, #FF7A5C 100%);
        color: #2C3E50 !important;
        border: none;
        border-radius: 25px;
        padding: 12px 30px;
        font-size: 16px;
        font-weight: bold;
        box-shadow: 0 4px 10px rgba(255, 122, 92, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #FF7A5C 0%, #FF6347 100%);
        box-shadow: 0 6px 15px rgba(255, 122, 92, 0.5);
        transform: translateY(-2px);
        color: #1A252F !important;
    }
    
    /* Warning box */
    .warning-box {
        background: #FFF3CD;
        border-left: 5px solid #FFC107;
        padding: 15px;
        border-radius: 8px;
        margin: 20px 0;
        color: #856404 !important;
    }
    
    .warning-box strong {
        color: #856404 !important;
    }
    
    /* Info cards */
    .info-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #4A90E2;
    }
    
    .info-card h4 {
        color: #2C3E50 !important;
        margin-bottom: 8px;
    }
    
    .info-card p {
        color: #5A6C7D !important;
        margin: 0;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #FFF5F0;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #FF9B85;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #FF7A5C;
    }
</style>
""", unsafe_allow_html=True)

# Load API keys
load_dotenv()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# State for conversation
class State(TypedDict):
    messages: Sequence[BaseMessage]

# Setup LLM and Tools
@st.cache_resource
def setup_agent():
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    
    if not GROQ_API_KEY:
        st.error("⚠️ GROQ_API_KEY not found! Add it to your .env file.")
        st.stop()
    
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.7
    )
    
    # Define tools
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    arxiv_tool = ArxivQueryRun()
    
    @tool
    def tavily_search(query: str) -> str:
        """Search the web for up-to-date info on skin cancer."""
        try:
            client = TavilyClient(api_key=TAVILY_API_KEY)
            response = client.search(query=query, search_depth="basic", max_results=3)
            results = [f"Title: {r['title']}\nSnippet: {r['content'][:200]}...\nURL: {r['url']}" 
                      for r in response["results"]]
            return "\n\n".join(results)
        except Exception as e:
            return f"Search failed: {str(e)}"
    
    ddg_search = DuckDuckGoSearchRun()
    tools = [wikipedia, arxiv_tool, tavily_search, ddg_search]
    llm_with_tools = llm.bind_tools(tools)
    
    # Agent function
    def agent(state: State):
        messages = state["messages"]
        last_msg = messages[-1].content.lower()
        
        # Check if it's a skin cancer related query
        is_skin_cancer = any(word in last_msg for word in 
                            ["skin cancer", "skincancer", "melanoma", "carcinoma", 
                             "types of skin cancer", "basal cell", "squamous", 
                             "research", "papers", "studies", "treatment", "symptoms",
                             "dermatology", "mole", "lesion", "biopsy", "skin",
                             "prevention", "sunscreen", "uv", "diagnosis"])
        
        if is_skin_cancer:
            response = llm_with_tools.invoke(messages)
            
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_results = []
                
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    args = tool_call["args"]
                    
                    try:
                        if "wikipedia" in tool_name.lower():
                            result = wikipedia.run(args.get("query", "skin cancer types"))
                            tool_results.append(f"📚 Wikipedia Result:\n{result[:800]}")
                        elif "arxiv" in tool_name.lower():
                            result = arxiv_tool.run(args.get("query", "skin cancer"))
                            tool_results.append(f"📄 ArXiv Research:\n{result[:800]}")
                        elif "tavily" in tool_name.lower():
                            result = tavily_search.invoke({"query": args.get("query", last_msg)})
                            tool_results.append(f"🔍 Web Search:\n{result[:800]}")
                        else:
                            result = ddg_search.run(args.get("query", last_msg))
                            tool_results.append(f"🌐 Search Result:\n{result[:800]}")
                    except Exception as e:
                        tool_results.append(f"⚠️ Tool error: {str(e)}")
                
                # Combine all tool results
                combined_results = "\n\n".join(tool_results)
                messages.append(AIMessage(content=combined_results))
                
                # Generate final response with context
                final_prompt = HumanMessage(content=f"Based on the above search results, please provide a comprehensive answer to: {messages[0].content}")
                messages.append(final_prompt)
                final_response = llm.invoke(messages)
                
                return {"messages": [final_response]}
            
            return {"messages": [response]}
        else:
            # Politely redirect off-topic queries
            redirect_message = AIMessage(content="""
I'm specifically designed to help with **skin cancer** related questions. 🏥

I can assist you with:
- 🔬 Types of skin cancer (melanoma, basal cell, squamous cell)
- 🩺 Symptoms and diagnosis
- 💊 Treatment options
- 🛡️ Prevention and sun safety
- 📚 Research papers and latest studies
- 🤔 Any concerns about skin lesions or moles

For other topics (like Python coding), you might want to use ChatGPT or Claude's general assistant!

**Would you like to ask me anything about skin cancer?** 😊
            """)
            return {"messages": [redirect_message]}
    
    # Build graph
    workflow = StateGraph(State)
    workflow.add_node("agent", agent)
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", END)
    return workflow.compile()

# Initialize agent
try:
    app = setup_agent()
except Exception as e:
    st.error(f"Error initializing agent: {str(e)}")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("### 🏥 About")
    st.markdown("""
    This AI assistant helps answer questions about:
    - **Skin cancer types** (melanoma, basal cell, squamous cell)
    - **Symptoms & diagnosis**
    - **Treatment options**
    - **Prevention tips**
    
    <style>
    .sidebar-text { color: white !important; }
    .sidebar-text li { color: white !important; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### ⚠️ Important")
    st.markdown("""
    <div class="warning-box">
    <strong>Medical Disclaimer:</strong><br>
    This chatbot provides educational information only. 
    Always consult healthcare professionals for medical advice.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📚 Quick Topics")
    topics = [
        "What are the types of skin cancer?",
        "Melanoma symptoms",
        "Skin cancer prevention",
        "Treatment options for basal cell carcinoma"
    ]
    for topic in topics:
        if st.button(topic, use_container_width=True, key=topic):
            st.session_state.pending_query = topic

# Main header
st.markdown("""
<div class="main-header">
    <h1>🏥 Skin Cancer AI Assistant</h1>
    <p>Your trusted companion for skin cancer information and support</p>
</div>
""", unsafe_allow_html=True)

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', 
                       unsafe_allow_html=True)
        else:
            # Use a container with custom class for bot messages
            st.markdown('<div class="bot-message-container">', unsafe_allow_html=True)
            st.markdown("🤖 **Assistant:**")
            st.markdown(message["content"])
            st.markdown('</div>', unsafe_allow_html=True)

# Handle pending query from sidebar
if "pending_query" in st.session_state:
    user_input = st.session_state.pending_query
    del st.session_state.pending_query
else:
    user_input = None

# Chat input
col1, col2 = st.columns([6, 1])
with col1:
    user_query = st.text_input(
        "Ask me anything about skin cancer...",
        key="user_input",
        placeholder="e.g., What are the early signs of melanoma?",
        label_visibility="collapsed"
    )
with col2:
    send_button = st.button("Send 📤", use_container_width=True)

# Process input
if (send_button and user_query) or user_input:
    query = user_input if user_input else user_query
    
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": query})
    st.session_state.messages.append(HumanMessage(content=query))
    
    # Show user message
    st.markdown(f'<div class="user-message">👤 {query}</div>', unsafe_allow_html=True)
    
    # Show loading
    with st.spinner("🔍 Analyzing your question..."):
        try:
            # Get response from agent
            response_text = ""
            for output in app.stream({"messages": st.session_state.messages}):
                for node_key, node_output in output.items():
                    if "messages" in node_output:
                        node_messages = node_output["messages"]
                        if node_messages:
                            last_response = node_messages[-1]
                            if hasattr(last_response, 'content') and last_response.content:
                                response_text = last_response.content
                                st.session_state.messages.append(last_response)
            
            if response_text:
                st.session_state.chat_history.append({"role": "bot", "content": response_text})
                # Render with markdown support
                st.markdown('<div class="bot-message-container">', unsafe_allow_html=True)
                st.markdown("🤖 **Assistant:**")
                st.markdown(response_text)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                error_msg = "Sorry, I couldn't generate a response. Please try again!"
                st.session_state.chat_history.append({"role": "bot", "content": error_msg})
                st.markdown(f'<div class="bot-message">🤖 {error_msg}</div>', 
                           unsafe_allow_html=True)
        
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            st.session_state.chat_history.append({"role": "bot", "content": error_msg})
            st.markdown(f'<div class="bot-message">🤖 {error_msg}</div>', 
                       unsafe_allow_html=True)
    
    st.rerun()

# Footer info
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="info-card">
        <h4>🔬 Powered by AI</h4>
        <p>Using advanced language models and medical databases</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="info-card">
        <h4>📖 Evidence-Based</h4>
        <p>Information from Wikipedia, ArXiv, and medical sources</p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="info-card">
        <h4>🔒 Privacy First</h4>
        <p>Your conversations are private and secure</p>
    </div>
    """, unsafe_allow_html=True)