import streamlit as st
import os
import tempfile
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler


from langchain.schema import AgentAction

import pandas as pd
import io
import tempfile
import json
import re       
        
# Import modularized components
from modules.data_preparation import prepare_dataframe
from modules.agent_tools import create_python_tool
from modules.query_processing import create_guardrail_chain, run_guardrail_loop, extract_query, extract_chat_history_from_string
            

class StreamlitChatCallbackHandler(BaseCallbackHandler):
    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        
        thought = action.log.split("Action")[0].strip()
        
        # with st.chat_message("assistant"):
        #     st.markdown("**🤔 Thought:**")
        #     st.markdown(action.log)
        
        with st.chat_message("assistant"):
            st.markdown("**🤔 Thought:**")
            #st.markdown(action.log)
            st.markdown(f"`{thought}`")

            st.markdown("**🔧 Action:**")
            st.markdown(f"`{action.tool}`")

            st.markdown("**🧾 Action Input:**")
            st.code(str(action.tool_input), language="python")

    def on_tool_end(self, output: str, **kwargs) -> None:
        with st.chat_message("assistant"):
            st.markdown("**👀 Observation:**")

            # Try to render tabular data if present
            try:
                df = pd.read_fwf(io.StringIO(output))
                if not df.empty and df.shape[1] > 1:
                    st.dataframe(df, use_container_width=True)
                else:
                    st.code(output)
            except Exception:
                st.code(output)            
            
# Set page configuration
st.set_page_config(
    page_title="Talk2Table",
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
    # model_name = st.selectbox(
    #     "Select Model",
    #     ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
    #     index=0
    # )
    
    # Clear buttons
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        st.session_state.memory_gr = ConversationBufferMemory(memory_key="chat_history", input_key="user_input", return_messages=True)
        st.success("Chat history cleared!")
    
    # if st.button("Start New Session"):
    #     st.session_state.messages = []
    #     st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    #     st.session_state.memory_gr = ConversationBufferMemory(memory_key="chat_history", input_key="user_input", return_messages=True)
    #     st.session_state.df = None
    #     st.success("New session started! Please upload a new CSV file.")

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
st.title("Talk2Table")

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
    llm_gr = ChatOpenAI(
        temperature=1,
        model_name="gpt-4o-mini"
    )
    
    # Set up the LLM
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo"
    )
    
    
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
    1. Leverage all information above to understand the user query and rephase it with clarity for next tool. 
    2. If the user query is ambiguous or unclear (e.g., refers to something not in columns), ask for clarification.
    3. Once the query is clear, rephrase it into a precise form for downstream analysis.
    4. If the query is already clear and relevant to the dataset, just rephrase it clearly.
    5. Make sure to not have word 'clarification' in the response if query is clear.

    Respond ONLY in one of the following formats:
    - If unclear:
    Clarification Needed: <your clarification question>
    - If clear:
    Rephrased Query: <your improved query>
    """
)

    
    
    # Create guardrail chain
    guardrail_chain, _ = create_guardrail_chain(llm_gr, guardrail_prompt)
    st.session_state.guardrail_chain = guardrail_chain
    
    # Initialize the agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # ZERO_SHOT_REACT_DESCRIPTION, #CONVERSATIONAL_REACT_DESCRIPTION
        verbose=True,
        memory=st.session_state.memory,
        
    )
    st.session_state.agent = agent
    
    container = st.container()
    callback = StreamlitChatCallbackHandler()
    
    
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
        
        with st.spinner("Processing query..."):
            if st.session_state.guardrail_chain:
                # Custom processing for Streamlit UI
                
                chat_history = extract_chat_history_from_string(str(st.session_state.memory_gr.buffer))
                
                # Run the guardrail chain
                inputs = {
                    "user_input": user_query,
                    "df_info": st.session_state.df_info_str,
                    "col_desc": st.session_state.col_desc_str,
                    "chat_history": chat_history
                }
                
                guardrail_response = st.session_state.guardrail_chain.run(**inputs)
                # Update memory
                st.session_state.memory_gr.save_context({"user_input": user_query}, {"output": guardrail_response})
                
                chat_history = extract_chat_history_from_string(str(st.session_state.memory_gr.buffer))
                
                #with st.chat_message("assistant"):
                        #st.markdown("**🤔 current memory:**")
                        #st.markdown(chat_history)
                        #st.markdown("**🤔 Raw:**")
                        #st.markdown(str(st.session_state.memory_gr.buffer))
                

                # Check if clarification is needed
                if "clarification" in guardrail_response.lower():
                    # Display assistant message for clarification
                    with st.chat_message("assistant"):
                        st.markdown(guardrail_response)
                    
                    # Add assistant message to history
                    st.session_state.messages.append({"role": "assistant", "content": guardrail_response})
                    
                else:
                    # Extract the query
                    final_query = extract_query(guardrail_response)
                    print(f"Final query: {final_query}")
                    # Update memory with the final query
                    # Run the agent
                    try:
                
                        result = st.session_state.agent.run(final_query, callbacks=[callback])
                        
                        # Display entire output by default
                        
                        st.markdown("#### Summarized Response")
                        st.markdown(result)

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
