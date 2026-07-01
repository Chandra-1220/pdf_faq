import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import faiss
import numpy as np
import torch

st.title("📄 PDF Q&A Bot")

# Load models
@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    model_name = "distilbert-base-cased-distilled-squad"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    return embedding_model, tokenizer, qa_model


embedding_model, tokenizer, qa_model = load_models()

# Upload PDF
pdf = st.file_uploader("Upload a PDF", type="pdf")

if pdf:

    reader = PdfReader(pdf)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text

    if text.strip() == "":
        st.error("No text found in the PDF.")
        st.stop()

    # Split text into chunks
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    # Generate embeddings
    embeddings = embedding_model.encode(chunks)

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    st.success(f"PDF processed successfully! ({len(chunks)} chunks)")

    # User question
    question = st.text_input("Ask a question")

    if st.button("Get Answer") and question:

        # Query embedding
        query_embedding = embedding_model.encode([question])

        # Retrieve top-3 chunks
        distances, indices = index.search(np.array(query_embedding), k=3)

        context = " ".join([chunks[i] for i in indices[0]])

        # Tokenize question and context
        inputs = tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        # Predict answer
        with torch.no_grad():
            outputs = qa_model(**inputs)

        start = torch.argmax(outputs.start_logits)
        end = torch.argmax(outputs.end_logits) + 1

        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(
                inputs["input_ids"][0][start:end]
            )
        )

        st.subheader("Answer")
        st.success(answer)

        with st.expander("Retrieved Context"):
            st.write(context)
