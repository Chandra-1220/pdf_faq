import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np

st.set_page_config(page_title="PDF Q&A Bot")

st.title("📄 PDF Q&A Bot")

# Load models
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-base"
    )

    return embed_model, generator


embed_model, generator = load_models()

# Upload PDF
pdf = st.file_uploader("Upload a PDF", type="pdf")

if pdf:

    reader = PdfReader(pdf)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text

    # Chunking
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    # Generate embeddings
    embeddings = embed_model.encode(chunks)

    # Create FAISS Index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    st.success("PDF uploaded successfully!")

    question = st.text_input("Ask a question")

    if st.button("Get Answer") and question:

        # Query embedding
        query_embedding = embed_model.encode([question])

        # Retrieve Top-3 Chunks
        distances, indices = index.search(
            np.array(query_embedding),
            k=3
        )

        context = " ".join(
            [chunks[i] for i in indices[0]]
        )

        prompt = f"""
Context:
{context}

Question:
{question}

Answer:
"""

        result = generator(
            prompt,
            max_new_tokens=100
        )

        st.subheader("Answer")

        st.success(result[0]["generated_text"])
