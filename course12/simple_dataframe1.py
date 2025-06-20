import pandas as pd
import os
from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

# ============ WHAT MAKES THIS AN AGENT ============
# An agent has 3 key components:
# 1. BRAIN (LLM) - Decides what to do
# 2. TOOLS - Actions it can take  
# 3. EXECUTOR - Runs the decision-making loop

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

# ============ AGENT COMPONENT 1: THE BRAIN ============
# The LLM that makes decisions about what to do
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# ============ AGENT COMPONENT 2: THE TOOLS ============
# The pandas agent automatically gets these tools:
# - python_repl_ast: Execute Python code
# - dataframe_info: Get info about the DataFrame
# - dataframe_head: Show first few rows
# - dataframe_describe: Get statistical summary

print("This is an AGENT because it has:")
print("1. BRAIN: LLM that reasons about what to do")
print("2. TOOLS: Python execution, DataFrame operations")
print("3. EXECUTOR: Decision loop that chooses tools")
print("\n" + "="*50 + "\n")

# ============ AGENT COMPONENT 3: THE EXECUTOR ============
# This creates the full agent with brain + tools + decision-making loop
agent = create_pandas_dataframe_agent(
    llm,          # THE BRAIN: Decides what action to take
    df,           # THE DATA: What the tools operate on
    verbose=True, # Shows the DECISION-MAKING PROCESS
    return_intermediate_steps=True,
    allow_dangerous_code=True
)

# ============ WHY IT'S AN AGENT (NOT JUST A CHAIN) ============
print("🤖 AGENT vs CHAIN:")
print("• CHAIN: Always does the same steps (prompt → LLM → output)")
print("• AGENT: Decides which tool to use based on the question")
print("• AGENT: Can use multiple tools in sequence")
print("• AGENT: Can reason about what to do next")
print("\n" + "="*50 + "\n")

# Example questions to ask the agent
questions = [
    "How many rows are in the dataframe?",
    "What is the average salary?",
    "Which person has the highest salary?",
    "How many people work in each department?",
    "What is the average age of people in Engineering?",
    "Show me all people from New York or London"
]

# Ask each question and show the AGENT DECISION-MAKING PROCESS
for i, question in enumerate(questions, 1):
    print(f"🔍 Question {i}: {question}")
    print("👀 Watch the agent's reasoning process:")
    try:
        response = agent.invoke(question)
        print(f"✅ Final Answer: {response['output']}")
        
        # Show what tools were used (this proves it's an agent!)
        if 'intermediate_steps' in response:
            print(f"🛠️  Tools used: {len(response['intermediate_steps'])} steps")
            for step_num, (action, result) in enumerate(response['intermediate_steps'], 1):
                print(f"   Step {step_num}: Used '{action.tool}' tool")
                
    except Exception as e:
        print(f"❌ Error: {e}")
    print("-" * 50)


