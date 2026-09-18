import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 1. UI Configuration & Extreme Custom CSS
st.set_page_config(page_title="Document AI", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Clean up default Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Center layout adjustments */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 95%;
    }
    
    /* Modern Typography for Greetings */
    .greeting-text {
        font-size: 2.5rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 0.2rem;
        color: #f1f5f9; /* Adjust to #1e293b if using light theme */
    }
    .sub-greeting {
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 3rem;
        color: #94a3b8;
    }
    
    /* Style suggestion buttons to look like chips */
    div[data-testid="stButton"] button {
        border-radius: 12px;
        border: 1px solid #334155;
        background-color: transparent;
        color: inherit;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease-in-out;
    }
    div[data-testid="stButton"] button:hover {
        border-color: #6366f1;
        background-color: rgba(99, 102, 241, 0.1);
        color: #6366f1;
    }

    /* Right column document upload box styling */
    .upload-box {
        border-radius: 12px;
        border: 1px dashed #475569;
        padding: 2rem;
        text-align: center;
        background-color: #0f172a;
    }
</style>
""", unsafe_allow_html=True)

# 2. Left Sidebar (Mimicking the History/Nav panel)
# 2. Left Sidebar (Clean Navigation)
with st.sidebar:
    st.subheader("NovaChat AI")
    
    # Optional: A button to start a fresh chat in the future
    st.button("➕ New Chat", use_container_width=True)
    
    st.divider()
    
    # Keep the reset button to clear the AI's memory
    if st.button("🗑️ Reset Memory", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.ui_messages = []
        st.rerun()
        
# 3. Core Logic Functions
@st.cache_resource
def get_vectorstore():
    # Placeholder for persistent DB
    return None

def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    os.remove(tmp_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(embedding_function=embeddings)
    
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vectorstore.add_documents(batch)
        
    return vectorstore

def create_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOllama(model="llama3.2")

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question formulate a standalone question. Do NOT answer the question, just reformulate it if needed."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a highly capable AI assistant. Answer based on the retrieved context below. Keep it clean and direct.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

# 4. Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []       
if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = []        
if "current_filename" not in st.session_state:
    st.session_state.current_filename = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# 5. Main 2-Column Layout (Chat Center | Document Right)
chat_col, doc_col = st.columns([3, 1], gap="large")

# --- RIGHT COLUMN: Document Management ---
with doc_col:
    st.markdown("### Upload Document")
    st.caption("Drag & drop or click to upload PDF")
    uploaded_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        if st.session_state.current_filename != uploaded_file.name:
            with st.spinner("Indexing..."):
                vectorstore = process_pdf(uploaded_file)
                st.session_state.rag_chain = create_rag_chain(vectorstore)
                st.session_state.current_filename = uploaded_file.name
                st.session_state.chat_history = []  
                st.session_state.ui_messages = []
            st.success("Ready!")
        
        st.markdown(f"**Active File:**\n`{uploaded_file.name}`")

# --- CENTER COLUMN: Chat Interface ---
with chat_col:
    # Show greeting and suggestion chips if chat is empty
    if len(st.session_state.ui_messages) == 0:
        st.markdown('<div class="greeting-text">Hello Chado,</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-greeting">Ask me anything. I\'m here to help.</div>', unsafe_allow_html=True)
        
        # Suggestion Chips Layout
        chip_col1, chip_col2 = st.columns(2)
        with chip_col1:
            st.button("📄 Summarize this document", use_container_width=True)
            st.button("💡 Brainstorm ideas", use_container_width=True)
        with chip_col2:
            st.button("⚡ Extract key points", use_container_width=True)
            st.button("📝 Help with a project", use_container_width=True)
    
    # Render Chat History
    else:
        for msg in st.session_state.ui_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "sources" in msg and msg["sources"]:
                    with st.expander("View Sources"):
                        for i, source in enumerate(msg["sources"]):
                            st.markdown(f"**Page {source['page']}**")
                            st.caption(source['text'])
                            if i < len(msg["sources"]) - 1:
                                st.divider()

    # Chat Input Box
    user_question = st.chat_input("Type a message...")

    if user_question:
        if st.session_state.rag_chain is None:
            st.error("Please upload a document first.")
        else:
            st.session_state.ui_messages.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.rag_chain.invoke({
                        "input": user_question,
                        "chat_history": st.session_state.chat_history
                    })
                    
                    ai_answer = response["answer"]
                    source_documents = response["context"]

                    extracted_sources = []
                    for doc in source_documents:
                        page_num = doc.metadata.get("page", 0) + 1 
                        extracted_sources.append({
                            "page": page_num,
                            "text": doc.page_content
                        })

                    st.markdown(ai_answer)
                    with st.expander("View Sources"):
                        for i, source in enumerate(extracted_sources):
                            st.markdown(f"**Page {source['page']}**")
                            st.caption(source['text'])
                            if i < len(extracted_sources) - 1:
                                st.divider()

            st.session_state.chat_history.extend([
                HumanMessage(content=user_question),
                AIMessage(content=ai_answer),
            ])
            st.session_state.ui_messages.append({
                "role": "assistant", 
                "content": ai_answer, 
                "sources": extracted_sources
            })
            st.rerun() # Force re-render to hide the initial greeting