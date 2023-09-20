import streamlit as st
from utils import Model
from langchain.llms import OpenAI

st.title("WebAsker - Ask me anything !")

st.sidebar.header("Train a new model")

st.sidebar.markdown("You can train a new model with data from website urls.")

urls = st.sidebar.text_area("URLs (one per line)")

OpenAPI_key = st.sidebar.text_input("OpenAI API Key", type="password")

train_button = st.sidebar.button("Train")

if train_button:
    model = Model(urls=urls.split("\n"), llm=OpenAI(openai_api_key=OpenAPI_key))
    model_trained = model.train()

    if model_trained:
        st.sidebar.success("Model trained !")
    # Q/A fields

        st.header("Ask me anything !")

        question = st.text_input("Question")
        ask_button = st.button("Ask")

        if ask_button:
            answer = model.answer(question)
            st.write(answer)
