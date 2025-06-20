from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

import os
from dotenv import load_dotenv
#print(os.getcwd())

# Expand ~ to full home directory path
dotenv_path = os.path.expanduser("~/.env")
# Load variables from .env file
load_dotenv(dotenv_path=dotenv_path)

# Access them using os.getenv or os.environ
api_key = os.getenv("OPENAI_API_KEY")

#print(f"OPENAI_API_KEY: {api_key}")

llm = ChatOpenAI()

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?"
)

chain = LLMChain(llm=llm, prompt=prompt)

response = chain.run("toothbrushes")
print(response)

