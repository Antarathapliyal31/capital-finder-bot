from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
import streamlit as st
st.title("Capital Finding Bot")
api=st.text_input("enter your OpenAI API key", type="password")
user_input=st.text_input("Enter a country name ")
prompt=PromptTemplate.from_template("What is the capital of {country} ?")
if api:
    os.environ["OPENAI_API_KEY"] = api
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0,max_tokens=20)
    chain=prompt | llm
    if st.button("Find the Capital"):
        if user_input:
            response = chain.invoke({"country": user_input})
            st.success(response.content)
        else:
            st.warning("Please enter a country name")
else:
    st.info("Enter a valid OpenAI API key to use the bot")
#temperature controls how random or creative the model’s responses are.
#0-Very focused, deterministic (same input = same output every time)
#0.5-Balanced — some creativity, still consistent
#0.8+-Very creative/random — great for storytelling or brainstorming
# Ollama = A free app to run powerful AI chatbots directly on your laptop. No internet, no API key, no cost.

