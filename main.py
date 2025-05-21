# main.py (previously csv_agent.py)
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import sweetviz as sv

# Import our modularized components using relative imports
import modules
from modules.data_preparation import prepare_dataframe
from modules.agent_tools import create_python_tool
from modules.query_processing import create_guardrail_chain, run_guardrail_loop, extract_chat_history_from_string
from modules.dataframe_analyzer import DataFrameAnalyzer, ColumnDescriptionParser, generate_dataset_report_for_llm


# Load environment variables
load_dotenv()

# Load dataframe and related information
CSV_FILE = "./data/customers-100.csv"
DESCRIPTION_FILE = "./data/df_desc.txt"
df, df_info_str, col_desc_str, globals_dict = prepare_dataframe(CSV_FILE, DESCRIPTION_FILE)

col_desc_str = str(generate_dataset_report_for_llm(df, col_desc_str, os.getenv("OPENAI_API_KEY"), verbose=True))

report = sv.analyze(df)

# Create tools for the agent
tools = create_python_tool(globals_dict, col_desc_str)

# Create guardrail prompt template
guardrail_prompt = PromptTemplate(
    input_variables=["user_input", "df_info", "col_desc", "chat_history"],
    template="""
    You are a bridge between query tool and user. Your job is to make sense of a user's input and make clear instruction for next tool about what user wants.

    The dataset has the following columns:
    {df_info}
    Based on the columns, you can infer which columns are relevant to the user query.
    
    Please refer to column descriptions for better clarity and how to make sense of the columns:
    {col_desc}

    Conversation so far we have with the user:
    {chat_history}
    You should consider this conversation to add context to the user query.
    
    The User query is:
    "{user_input}"

    Instructions:
    1. If the current query refers to something comparative (e.g., "lowest", "most", "that", "then", "it"), use the chat history to determine what the user is referring to.
    2. For example, if user previously asked which team won highest number of matches, and now asking "then what was the score of that team", you should understand that user is asking about the score of the team with highest number of matches.
    3. If the user query is ambiguous or unclear (e.g., refers to something not in columns), ask for clarification.
    4. Once the ask in the query is clear, rephrase it into a precise form for downstream analysis.
    5. If the query is already clear and relevant to the dataset, just rephrase it clearly.
    6. Make sure to not have word 'clarification' in the response if query is clear.
    
    Respond ONLY in one of the following formats:
    - If unclear:
    Clarification Needed: <your clarification question>
    - If clear:
    Rephrased Query: <your improved query>
    """
)

# Set up the LLM
llm_gr = ChatOpenAI(
    temperature=1,
    model_name= "gpt-4o-mini"
)

llm = ChatOpenAI(
    temperature=0,
    model_name= "gpt-3.5-turbo"
)
# Create guardrail chain
guardrail_chain, memory_gr = create_guardrail_chain(llm_gr, guardrail_prompt)

# Set up the agent memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, #ZERO_SHOT_REACT_DESCRIPTION #CONVERSATIONAL_REACT_DESCRIPTION
    verbose=True,
    memory=memory,
    handle_parsing_errors=True,
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
            #extract_chat_history_from_string(str(memory.chat_memory.messages))
            
            final_query = run_guardrail_loop(
                clarifier_chain=guardrail_chain,
                original_input=query,
                df_info=df_info_str,
                col_desc=col_desc_str,
                memory_gr=memory_gr
            )
            #print("memory_gr:", memory_gr.chat_memory.messages)
            #print('\n\n')
            #print("memory:", memory.chat_memory.messages)
            #print('\n\n')
            
            print("🧠 Final Query:", final_query)
            result = agent.run(final_query)
            print(f"🧠 Answer: {result}\n")
        except Exception as e:
            print(f"❌ Error: {e}")
