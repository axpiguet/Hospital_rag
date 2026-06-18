import streamlit as st
from vector_store import build_index
from rag import answer_question

st.set_page_config(page_title="Hospital RAG Demo", page_icon="🏥")
st.title("🏥 Hospital Knowledge Assistant (Demo)")

with st.sidebar:
    if st.button("🔄 Rebuild index"):
        n = build_index(reset=True)
        st.success(f"Indexed {n} chunks.")

question = st.text_input("Ask a question:")
if st.button("Ask") and question.strip():
    with st.spinner("Thinking..."):
        result = answer_question(question)
    st.subheader("Answer")
    st.write(result["answer"])
    st.subheader("Sources")
    st.write(", ".join(result["sources"]))
    with st.expander("Retrieved chunks"):
        for h in result["hits"]:
            st.markdown(f"**{h['source']}** (distance: {h['distance']:.4f})")
            st.text(h["text"])
            