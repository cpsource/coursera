from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

import os
from dotenv import load_dotenv

# Expand ~ to full home directory path
dotenv_path = os.path.expanduser("~/.env")
# Load variables from .env file
load_dotenv(dotenv_path=dotenv_path)

# Access them using os.getenv or os.environ
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI()

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?"
)

# Modern approach: chain the prompt and LLM directly
chain = prompt | llm
print(f"chain = {chain}")

# Use invoke() instead of run()
response = chain.invoke({"product": "toothbrushes"})
print(response.content)
