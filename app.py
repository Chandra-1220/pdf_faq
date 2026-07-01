import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np

st.title("📄 PDF Q&A Bot")

# Load models
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    qa_model = pipeline("question-answering")
    return embed_model, qa_model

embed_model, qa_model = load_models()

# Upload PDF
pdf = st.file_uploader("Upload PDF", type="pdf")

if pdf:

    # Read PDF
    reader = PdfReader(pdf)
    text = ""

    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()

    # Split into chunks
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    # Generate embeddings
    embeddings = embed_model.encode(chunks)

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    st.success("PDF uploaded successfully!")

    # Ask question
    question = st.text_input("Ask a question")

    if st.button("Get Answer") and question:

        # Query embedding
        query_embedding = embed_model.encode([question])

        # Retrieve top 3 chunks
        distances, indices = index.search(np.array(query_embedding), k=3)

        context = " ".join([chunks[i] for i in indices[0]])

        # Generate answer
        answer = qa_model(question=question, context=context)

        st.subheader("Answer")
        st.write(answer["answer"])
