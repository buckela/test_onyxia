import streamlit as st

st.set_page_config(page_title="test1", page_icon="O")

st.title("test streamlit onyxia")
st.write("test")

case = st.text_input("case", "0")
if st.button("send"):
    st.success(f"case : {case}\n")

valeur = st.slider("test", 0, 1, 2)
st.progress(valeur/2)

