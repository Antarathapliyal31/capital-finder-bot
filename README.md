# capital-finder-bot
A beginner-friendly chatbot built with LangChain and Ollama (Mistral model) that answers capital city questions based on user input.
It is powered by a simple Streamlit web interface so that anyone can try it locally.

Build using:
- [LangChain](https://www.langchain.com/)
- [Ollama](https://ollama.com/) (running Mistral model locally)
- [Streamlit](https://streamlit.io) – to create a lightweight and interactive web UI 
- Python
  
Features:
- Uses LangChain prompt templates
- Powered by Mistral (via Ollama)
- No API key or internet required
- Local, free, and beginner-friendly
- Interactive web interface with **Streamlit**

How to Run
1. Install dependencies:
   pip install streamlit langchain-community langchain-ollama
2. Download Ollama
3. Run Ollama and pull the mistral model:
   ollama run mistral
4. Run the app- streamlit run capital_bot.py

This is built as part of my journey learning how to create LLM-based assistants using LangChain and local models like Mistral.



