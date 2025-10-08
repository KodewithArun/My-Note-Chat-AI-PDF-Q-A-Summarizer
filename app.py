import streamlit as st
import os
import shutil
from dotenv import load_dotenv

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableSequence

# PDF Parsing
from llama_parse import LlamaParse

# Load environment variables
load_dotenv()
LLAMA_API_KEY = os.getenv("LLAMA_PARSE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Directories
UPLOAD_DIR = "documents/uploaded_file"
PARSED_DIR = "documents/parsed_md"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PARSED_DIR, exist_ok=True)

st.set_page_config(page_title="My Note Chat", layout="wide")

st.title("My Note Chat")
st.write("Upload a PDF to either ask questions or get a full summary. Uses LlamaParse, Groq, and LangChain.")

uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

mode = st.radio("Choose mode:", ["Question Answering", "Summarization"], horizontal=True)

if uploaded_file:
    pdf_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success(f"✅ File uploaded: {uploaded_file.name}")

    # Parse PDF
    if "parsed_docs" not in st.session_state or st.session_state["last_file"] != uploaded_file.name:
        with st.spinner("Parsing PDF with LlamaParse..."):
            parser = LlamaParse(
                api_key=LLAMA_API_KEY,
                result_type="markdown",
                verbose=True,
                system_prompt="""You are a document parser. Extract the content into clean, structured markdown format.
                - Preserve headings, subheadings, paragraphs clearly.
                - Convert tables into markdown tables.
                - Represent images with markdown image syntax.
                - Avoid extra line breaks or broken markdown syntax.
                """
            )
            docs = parser.load_data(pdf_path)
            st.session_state["parsed_docs"] = docs
            st.session_state["last_file"] = uploaded_file.name
    else:
        docs = st.session_state["parsed_docs"]

    # Save markdown
    md_file_path = os.path.join(PARSED_DIR, uploaded_file.name.replace(".pdf", ".md"))
    with open(md_file_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(doc.text + "\n\n")

    # Convert to LangChain docs
    lc_docs = [
        Document(page_content=d.text, metadata={"source": pdf_path, "page": i+1})
        for i, d in enumerate(docs)
    ]

    # Setup Parent Retriever
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(embedding_function=embeddings, persist_directory="chroma_db")
    doc_store = InMemoryStore()

    parent_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=doc_store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )

    parent_retriever.add_documents(lc_docs)

    # Setup LLM
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)

    # QA prompt
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a helpful and fact-based QA assistant.\n\n"
            "Use ONLY the information from the context below to answer the question.\n"
            "If the context does not clearly contain the answer, reply exactly with:\n"
            "\"Insufficient information in the provided context.\"\n\n"
            "Be clear, concise, and strictly factual.\n"
            "Do not add assumptions or outside knowledge.\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Answer:"
        )
    )

    # Helper to extract text
    def extract_text(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    retriever_chain = RunnableLambda(lambda x: parent_retriever.invoke(x["question"])) | RunnableLambda(extract_text)

    qa_chain = RunnableSequence(
        RunnableParallel({
            "context": retriever_chain,
            "question": RunnableLambda(lambda x: x["question"])
        })
        | qa_prompt
        | llm
    )

    if mode == "Question Answering":
        question = st.text_input("Ask your question based on the document:")
        if st.button("Get Answer"):
            if question.strip():
                with st.spinner("Searching and generating answer..."):
                    result = qa_chain.invoke({"question": question})
                st.subheader("Answer:")
                st.write(result.content)
            else:
                st.warning("Please enter a question.")

    else:  # Summarization
        with st.spinner("Summarizing document..."):
            child_docs = []
            for parent in lc_docs:
                child_docs.extend(parent_retriever.child_splitter.split_text(parent.page_content))

            chunk_summarize_prompt = PromptTemplate(
                input_variables=["context"],
                template=(
                    "Summarize the following text into clear, concise paragraphs. "
                    "Focus on the main ideas and present them in a natural, easy-to-read style. "
                    "Do not use bullet points, headers, or repeated phrases. "
                    "Format the summary as well-structured paragraphs with proper line breaks and spacing.\n\n"
                    "Text:\n{context}\n\n"
                    "Summary:"
                )
            )

            chunk_summaries = []
            for c in child_docs:
                summarize_chain_chunk = RunnableSequence(
                    RunnableParallel({
                        "context": RunnableLambda(lambda x: c)
                    })
                    | chunk_summarize_prompt
                    | llm
                )
                result = summarize_chain_chunk.invoke({"topic": "ignored"})
                chunk_summaries.append(result.content.strip())

            final_summary = "\n".join([
                line for chunk in chunk_summaries
                for line in chunk.split("\n")
                if line.strip() != ""
            ])

        st.subheader("Summary:")
        st.text_area("Document Summary", final_summary, height=1000, label_visibility="collapsed")
