# capital-finder-bot
A beginner-friendly chatbot built with **LangChain** and **OpenAI's GPT-3.5** that answers capital city questions based on user input.
It is powered by a simple Streamlit web interface so that anyone can try it.

Build using:
- [LangChain](https://www.langchain.com/)
- [OpenAI API](https://platform.openai.com/) – using GPT-3.5 for natural language generation
- [Streamlit](https://streamlit.io) – to create a lightweight and interactive web UI 
- Python
  
Features:
- Uses LangChain prompt templates
- Powered by **GPT-3.5** through the OpenAI API
- Use of API keys, prompt chaining, and model integration
- Interactive web interface with **Streamlit**

How to Run
1. Install dependencies:
   pip install streamlit langchain-community langchain-openai openai
2. Download Ollama
3. Sign in at https://platform.openai.com/account/api-keys & copy your API key.
4. Run the app - streamlit run capital_bot.py

This bot is part of my learning journey in building LLM-based apps using LangChain and OpenAI.
I wanted to explore:
1. How prompt templates work
2. How to integrate LLMs into Streamlit apps
3. How to accept user inputs and securely use API keys
