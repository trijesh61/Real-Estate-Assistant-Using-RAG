import streamlit as st
from rag import process_urls,generate_answer




st.title("Real Estate Research Tool")

url1=st.sidebar.text_input("URL 1")
url2=st.sidebar.text_input("URL 2")
url3=st.sidebar.text_input("URL 3")

placeholder=st.empty()
process_url_button=st.sidebar.button("Process URLs")
if process_url_button:
    urls = [url for url in (url1, url2, url3) if url is not None]
    if len(urls) == 0:
        placeholder.text("You must provide at least one valid url")
    else:
        for status in process_urls(urls):
            placeholder.text(status)