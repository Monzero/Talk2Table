# main.py (previously csv_agent.py)
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Import our modularized components using relative imports
import modules
from modules.data_preparation import prepare_dataframe
from modules.agent_tools import create_python_tool
from modules.query_processing import create_guardrail_chain, run_guardrail_loop

# Load environment variables
load_dotenv()

# Load dataframe and related information
CSV_FILE = "./data/customers-100.csv"
DESCRIPTION_FILE = "./data/df_desc.txt"
df, df_info_str, col_desc_str, globals_dict = prepare_dataframe(CSV_FILE, DESCRIPTION_FILE)

# Create tools for the agent
tools = create_python_tool(globals_dict, col_desc_str)

# Create guardrail prompt template
guardrail_prompt = PromptTemplate(
    input_variables=["user_input", "df_info", "col_desc", "chat_history"],
    template="""
    You are a query assessment agent. Your job is to decide if a user's query can be answered using the given dataset.

    The dataset has the following columns:
    {df_info}

    Column descriptions:
    {col_desc}

    Conversation so far:
    {chat_history}

    User query:
    "{user_input}"

    Instructions:
    1. If the user query is ambiguous or unclear (e.g., refers to something not in columns), ask for clarification.
    2. Once the query is clear, rephrase it into a precise form for downstream analysis.
    3. If the query is already clear and relevant to the dataset, just rephrase it clearly.
    4. Make sure to not have word 'clarification' in the response if query is clear.

    Respond ONLY in one of the following formats:
    - If unclear:
    Clarification Needed: <your clarification question>
    - If clear:
    Rephrased Query: <your improved query>
    """
)

# Set up the LLM
llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo"
)

# Create guardrail chain
guardrail_chain, memory_gr = create_guardrail_chain(llm, guardrail_prompt)

# Set up the agent memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
)

# Interactively test the agent
if __name__ == "__main__":
    print(f"\n✅ Loaded '{CSV_FILE}' with shape {df.shape}. DataFrame is available as 'df'.\n")
    memory.clear()
    memory_gr.clear()
    while True:
        try:
            query = input("💬 Ask a question about the CSV (or type 'exit'): ")
            if query.lower() in ["exit", "quit"]:
                break
            final_query = run_guardrail_loop(
                clarifier_chain=guardrail_chain,
                original_input=query,
                df_info=df_info_str,
                col_desc=col_desc_str,
                memory_gr=memory_gr
            )
            print("🧠 Final Query:", final_query)
            result = agent.run(final_query)
            print(f"🧠 Answer: {result}\n")
        except Exception as e:
            print(f"❌ Error: {e}")
