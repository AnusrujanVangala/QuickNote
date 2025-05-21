import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader, BSHTMLLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# === Sidebar ===
with st.sidebar:
    st.title("ℹ️ About")
    st.markdown("""
    **Academic RAG Assistant**  
    Upload a document and ask questions or generate notes/quizzes using Llama3-70B.
    - Supported: PDF, TXT, Markdown, HTML
    - Modes: Simple, Formal, In-depth, Quiz, Notes
    """)
    st.markdown("---")
    st.markdown("**Model:** llama3-70b-8192 via Groq")
    st.markdown("**Author:** Your Name")

# === Document Loader Function ===
def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    elif ext == ".md":
        loader = UnstructuredMarkdownLoader(file_path)
    elif ext in [".html", ".htm"]:
        loader = BSHTMLLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader.load()

# === Vector Store Creator ===
def create_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    texts = [doc.page_content for doc in docs]
    vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings)
    return vectorstore

# === Groq LLM Setup ===
llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# === Prompt Templates ===
simple_prompt = PromptTemplate.from_template("""
Answer the question in simple terms using the context below.
Context: {context}
Question: {question}
""")

formal_prompt = PromptTemplate.from_template("""
Answer the following question in a formal tone, in around 30 lines.
Use the provided context:
Context: {context}
Question: {question}
""")

in_depth_prompt = PromptTemplate.from_template("""
Provide an in-depth analysis of the topic based on the following content:
Context: {context}
Topic: {question}
""")

quiz_prompt = PromptTemplate.from_template("""
Generate {num_questions} quiz questions with answers based on the following content:
{context}
""")

notes_prompt = PromptTemplate.from_template("""
Generate up to 10 concise bullet point notes about the following topic:
{context}
""")

# === QA Chain Builder ===
def get_qa_chain(vectorstore, prompt):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return chain

# === Query Handler ===
def rag_query(vectorstore, question, mode):
    if mode == "simple":
        chain = get_qa_chain(vectorstore, simple_prompt)
    elif mode == "formal":
        chain = get_qa_chain(vectorstore, formal_prompt)
    elif mode == "indepth":
        chain = get_qa_chain(vectorstore, in_depth_prompt)
    elif mode.startswith("quiz"):
        num = int(mode.split("_")[1])
        prompt = quiz_prompt.partial(num_questions=str(num))
        chain = get_qa_chain(vectorstore, prompt)
    elif mode == "notes":
        chain = get_qa_chain(vectorstore, notes_prompt)
    else:
        raise ValueError("Invalid mode selected.")
    
    result = chain.invoke({"query": question})
    return result["result"], result.get("source_documents", [])

# === Streamlit UI ===
st.title("📚 Academic RAG Assistant")
uploaded_file = st.file_uploader("Upload a document (.pdf, .txt, .md, .html)", type=["pdf", "txt", "md", "html"])

if uploaded_file:
    temp_path = f"temp_{uploaded_file.name}"
    try:
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        with st.spinner("Processing document..."):
            docs = load_document(temp_path)
            vectorstore = create_vector_store(docs)
    except Exception as e:
        st.error(f"Error loading document: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        st.stop()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    question = st.text_input("Enter your question/topic:")
    mode = st.selectbox("Choose output style:", [
        "simple", "formal", "indepth", "quiz_10", "quiz_20", "notes"
    ])

    if st.button("Generate Response"):
        if not question.strip():
            st.warning("Please enter a question or topic.")
        else:
            with st.spinner("Generating response..."):
                try:
                    response, sources = rag_query(vectorstore, question, mode)
                    st.markdown("### ✅ Response")
                    st.write(response)
                    if sources:
                        with st.expander("Show source documents"):
                            for i, doc in enumerate(sources, 1):
                                st.markdown(f"**Source {i}:**")
                                st.write(doc.page_content[:1000] + ("..." if len(doc.page_content) > 1000 else ""))
                except Exception as e:
                    st.error(f"Error during RAG query: {e}")
else:
    st.info("Please upload a document to get started.")