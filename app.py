import streamlit as st
import os
import tempfile
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


# Import modularized components
from modules.data_preparation import prepare_dataframe
from modules.agent_tools import create_python_tool
from modules.query_processing import create_guardrail_chain, run_guardrail_loop, extract_query

# Set page configuration
st.set_page_config(
    page_title="CSV Agent Chat",
    page_icon="📊",
    layout="wide",
)

# Sidebar for file uploads and settings
with st.sidebar:
    st.title("CSV Agent Settings")
    
    # File uploads
    uploaded_csv = st.file_uploader("Upload CSV file", type="csv")
    uploaded_desc = st.file_uploader("Upload column descriptions (optional)", type="txt")
    
    # Model selection
    model_name = st.selectbox(
        "Select Model",
        ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
        index=0
    )
    
    # Clear buttons
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        st.session_state.memory_gr = ConversationBufferMemory(memory_key="chat_history", input_key="user_input", return_messages=True)
        st.success("Chat history cleared!")
    
    if st.button("Start New Session"):
        st.session_state.messages = []
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        st.session_state.memory_gr = ConversationBufferMemory(memory_key="chat_history", input_key="user_input", return_messages=True)
        st.session_state.df = None
        st.success("New session started! Please upload a new CSV file.")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

if "memory_gr" not in st.session_state:
    st.session_state.memory_gr = ConversationBufferMemory(memory_key="chat_history", input_key="user_input", return_messages=True)

if "df" not in st.session_state:
    st.session_state.df = None

if "df_info_str" not in st.session_state:
    st.session_state.df_info_str = ""

if "col_desc_str" not in st.session_state:
    st.session_state.col_desc_str = ""

if "globals_dict" not in st.session_state:
    st.session_state.globals_dict = {}

if "agent" not in st.session_state:
    st.session_state.agent = None

if "guardrail_chain" not in st.session_state:
    st.session_state.guardrail_chain = None

# Main content area
st.title("CSV Agent Chat")

# Load API key from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API key not found in .env file. Please add it to continue.")
    st.stop()

# Process uploaded files
if uploaded_csv:
    # Save uploaded CSV to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_csv:
        tmp_csv.write(uploaded_csv.getvalue())
        csv_path = tmp_csv.name
    
    # Save uploaded description to temporary file if provided
    desc_path = None
    if uploaded_desc:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_desc:
            tmp_desc.write(uploaded_desc.getvalue())
            desc_path = tmp_desc.name
            
    # Set OpenAI API Key
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Load dataframe and related information
    df, df_info_str, col_desc_str, globals_dict = prepare_dataframe(csv_path, desc_path)
    
    # Store in session state
    st.session_state.df = df
    st.session_state.df_info_str = df_info_str
    st.session_state.col_desc_str = col_desc_str 
    st.session_state.globals_dict = globals_dict
    
    # Set up the LLM
    llm = ChatOpenAI(
        temperature=0,
        model_name=model_name
    )
    
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
    
    # Create guardrail chain
    guardrail_chain, _ = create_guardrail_chain(llm, guardrail_prompt)
    st.session_state.guardrail_chain = guardrail_chain
    
    # Initialize the agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=st.session_state.memory,
    )
    st.session_state.agent = agent
    
    # Clean up temporary files
    if desc_path:
        os.unlink(desc_path)
    os.unlink(csv_path)
    
    st.success(f"✅ Loaded CSV with shape {df.shape}")

# Display data preview if available
if st.session_state.df is not None:
    with st.expander("Preview Data"):
        st.dataframe(st.session_state.df.head(10))
        
    # Display column descriptions if available
    if st.session_state.col_desc_str:
        with st.expander("Column Descriptions"):
            st.text(st.session_state.col_desc_str)

# Display the conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if st.session_state.df is not None:
    user_query = st.chat_input("Ask a question about your CSV data...")
    
    if user_query:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Process with guardrail
        with st.spinner("Processing query..."):
            if st.session_state.guardrail_chain:
                # Custom processing for Streamlit UI
                inputs = {
                    "user_input": user_query,
                    "df_info": st.session_state.df_info_str,
                    "col_desc": st.session_state.col_desc_str,
                    "chat_history": str(st.session_state.memory_gr.buffer)
                }
                
                guardrail_response = st.session_state.guardrail_chain.run(**inputs)
                
                # Check if clarification is needed
                if "clarification needed" in guardrail_response.lower():
                    # Display assistant message for clarification
                    with st.chat_message("assistant"):
                        st.markdown(guardrail_response)
                    
                    # Add assistant message to history
                    st.session_state.messages.append({"role": "assistant", "content": guardrail_response})
                    
                    # Update memory
                    st.session_state.memory_gr.save_context({"user_input": user_query}, {"output": guardrail_response})
                else:
                    # Extract the query
                    final_query = extract_query(guardrail_response)
                    
                    # Run the agent
                    try:
                        result = st.session_state.agent.run(final_query)
                        
                        # Display assistant message
                        with st.chat_message("assistant"):
                            st.markdown(result)
                        
                        # Add assistant message to history
                        st.session_state.messages.append({"role": "assistant", "content": result})
                        
                    except Exception as e:
                        error_message = f"Error: {str(e)}"
                        
                        # Display error message
                        with st.chat_message("assistant"):
                            st.error(error_message)
                        
                        # Add error message to history
                        st.session_state.messages.append({"role": "assistant", "content": error_message})
else:
    st.info("Please upload a CSV file to start chatting!")

# Footer
st.markdown("---")
st.caption("CSV Agent Chat - A tool for analyzing CSV data using natural language")
st.markdown("Built with ♥️ by Monil Shah")