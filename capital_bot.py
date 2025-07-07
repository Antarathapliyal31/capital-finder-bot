from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama

prompt=PromptTemplate.from_template("What is the capital of {country} ?")
#temperature controls how random or creative the model’s responses are.
#0-Very focused, deterministic (same input = same output every time)
#0.5-Balanced — some creativity, still consistent
#0.8+-Very creative/random — great for storytelling or brainstorming
# Ollama = A free app to run powerful AI chatbots directly on your laptop. No internet, no API key, no cost.
llm = Ollama(model="mistral", temperature=0,num_predict=26)
#Earlier I was not using temperature and num_predict parameters, it was taking 5.30 seconds, after using them it reduced the response time to 1.26 seconds)
chain = prompt | llm
user_input=input("Enter a country name")
response=chain.invoke({"country":user_input})
print("The answer to your question is",response)

