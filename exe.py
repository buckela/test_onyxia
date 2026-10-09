import streamlit as st

st.set_page_config(page_title="test1", page_icon="O")

#st.title("test streamlit onyxia")
#st.write("test")

name = st.text_input("name", "0")
if st.button("log"):
    case = st.text_input("case", "0")
    if st.button("send"):
        st.success(f"case : {case}\n")




