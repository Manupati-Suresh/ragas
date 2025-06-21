
import streamlit as st
from rag_pipeline import RAGPipeline

@st.cache_resource
def load_rag_pipeline():
    return RAGPipeline()

rag_pipeline = load_rag_pipeline()

st.title("RAG-Based Semantic Quote Retrieval and Structured QA")
st.write("Enter a query to find relevant quotes and get structured answers.")

query = st.text_input("Your Query:", "quotes about love")

if query:
    st.write("### Retrieved Quotes:")
    answer = rag_pipeline.answer_query(query)
    st.write(answer)


