from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
import streamlit as st
st.title("Capital Finding Bot")
user_input=st.text_input("Enter a country name ")
prompt=PromptTemplate.from_template("What is the capital of {country} ?")

#temperature controls how random or creative the model’s responses are.
#0-Very focused, deterministic (same input = same output every time)
#0.5-Balanced — some creativity, still consistent
#0.8+-Very creative/random — great for storytelling or brainstorming
# Ollama = A free app to run powerful AI chatbots directly on your laptop. No internet, no API key, no cost.
llm = OllamaLLM(model="mistral", temperature=0, num_predict=20)
chain = prompt | llm
if st.button("Find the Capital"):
    if user_input:
        response=chain.invoke({"country":user_input})
        st.success(response)
    else:
        st.warning("Please enter a country name")



