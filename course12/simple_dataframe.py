import pandas as pd
import os
from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

# Load environment variables
dotenv_path = os.path.expanduser("~/.env")
load_dotenv(dotenv_path=dotenv_path)

# Create sample data instead of reading from CSV
# This way you can run the example without needing an external file
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'Age': [25, 30, 35, 28, 32],
    'City': ['New York', 'London', 'Paris', 'Tokyo', 'Sydney'],
    'Salary': [50000, 60000, 70000, 55000, 65000],
    'Department': ['Engineering', 'Marketing', 'Engineering', 'HR', 'Marketing']
}

# Create DataFrame
df = pd.DataFrame(data)
print("Sample DataFrame:")
print(df)
print("\n" + "="*50 + "\n")

# Initialize the language model
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# Create the pandas DataFrame agent
agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,  # Shows the agent's reasoning process
    return_intermediate_steps=True,
    allow_dangerous_code=True  # Required for pandas operations
)

# Example questions to ask the agent
questions = [
    "How many rows are in the dataframe?",
    "What is the average salary?",
    "Which person has the highest salary?",
    "How many people work in each department?",
    "What is the average age of people in Engineering?",
    "Show me all people from New York or London"
]

# Ask each question
for i, question in enumerate(questions, 1):
    print(f"Question {i}: {question}")
    try:
        response = agent.invoke(question)
        print(f"Answer: {response['output']}")
    except Exception as e:
        print(f"Error: {e}")
    print("-" * 50)

