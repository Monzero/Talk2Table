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
#import sweetviz as sv

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
from modules.dataframe_analyzer import DataFrameAnalyzer, ColumnDescriptionParser, generate_dataset_report_for_llm

            

class StreamlitChatCallbackHandler(BaseCallbackHandler):
    # def on_chain_start(self, serialized: dict, inputs: dict, **kwargs) -> None:
    #     with st.chat_message("assistant"):
    #         st.markdown("🔄 **Chain started**")
    #         st.json(inputs)
     
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
    
    # def on_llm_end(self, response, **kwargs) -> None:
    #     with st.chat_message("assistant"):
    #         st.markdown("📝 **LLM Response:**")
    #         st.markdown(response.generations[0][0].text)


    def on_chain_end(self, outputs: dict, **kwargs) -> None:
        with st.chat_message("assistant"):
            st.markdown("✅ **Chain completed**")
            st.json(outputs)


def get_sample_datasets():
    """
    Get available sample datasets from the sample_data folder.
    
    Returns:
        dict: Dictionary with dataset names as keys and file paths as values
    """
    sample_data_folder = "sample_data"
    datasets = {}
    
    if os.path.exists(sample_data_folder):
        csv_files = [f for f in os.listdir(sample_data_folder) if f.endswith('.csv')]
        for csv_file in csv_files:
            dataset_name = os.path.splitext(csv_file)[0]
            csv_path = os.path.join(sample_data_folder, csv_file)
            
            # Look for corresponding description file
            desc_file = f"{dataset_name}_desc.txt"
            desc_path = os.path.join(sample_data_folder, desc_file)
            if not os.path.exists(desc_path):
                desc_path = None
            
            datasets[dataset_name] = {
                'csv_path': csv_path,
                'desc_path': desc_path,
                'display_name': dataset_name.replace('_', ' ').title()
            }
    
    return datasets


def load_sample_dataset(dataset_info):
    """
    Load a sample dataset and return the paths.
    
    Args:
        dataset_info (dict): Dataset information containing paths
        
    Returns:
        tuple: (csv_path, desc_path)
    """
    return dataset_info['csv_path'], dataset_info['desc_path']

    
# Set page configuration
st.set_page_config(
    page_title="Talk2Table",
    page_icon="🦁",
    layout="wide",
)

# Sidebar for file uploads and settings
with st.sidebar:
    st.title("Talk2table Settings")
    
    # Data source selection
    st.subheader("Data Source")
    data_source = st.radio(
        "Choose your data source:",
        ["Upload your own files", "Use sample data"],
        index=0
    )
    
    csv_path = None
    desc_path = None
    
    if data_source == "Upload your own files":
        # File uploads
        st.subheader("File Upload")
        uploaded_csv = st.file_uploader("Upload CSV file", type="csv")
        uploaded_desc = st.file_uploader("Upload column descriptions", type="txt")
        
        if uploaded_csv:
            # Save uploaded CSV to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_csv:
                tmp_csv.write(uploaded_csv.getvalue())
                csv_path = tmp_csv.name
            
            # Save uploaded description to temporary file if provided
            if uploaded_desc:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_desc:
                    tmp_desc.write(uploaded_desc.getvalue())
                    desc_path = tmp_desc.name
    
    else:  # Use sample data
        st.subheader("Sample Datasets")
        sample_datasets = get_sample_datasets()
        
        if sample_datasets:
            dataset_options = list(sample_datasets.keys())
            dataset_display_names = [sample_datasets[key]['display_name'] for key in dataset_options]
            
            selected_dataset = st.selectbox(
                "Select a sample dataset:",
                options=dataset_options,
                format_func=lambda x: sample_datasets[x]['display_name'],
                index=0
            )
            
            if selected_dataset:
                csv_path, desc_path = load_sample_dataset(sample_datasets[selected_dataset])
                st.success(f"✅ Selected: {sample_datasets[selected_dataset]['display_name']}")
                
                # Show dataset info
                if os.path.exists(csv_path):
                    try:
                        sample_df = pd.read_csv(csv_path)
                        st.info(f"📈 Dataset shape: {sample_df.shape[0]} rows, {sample_df.shape[1]} columns")
                    except Exception as e:
                        st.error(f"Error reading dataset: {str(e)}")
        else:
            st.warning("⚠️ No sample datasets found in the 'sample_data' folder.")
            st.info("To add sample datasets, create a 'sample_data' folder and add CSV files with optional description files (filename_desc.txt).")
    
    # Model selection (commented out in original)
    # model_name = st.selectbox(
    #     "Select Model",
    #     ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
    #     index=0
    # )
    
    # Clear buttons
    st.subheader("Session Management")
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

# Process files (either uploaded or sample)
if csv_path:
    # Set OpenAI API Key
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Load dataframe and related information
    df, df_info_str, col_desc_str, globals_dict = prepare_dataframe(csv_path, desc_path)
    
    col_desc_str = str(generate_dataset_report_for_llm(df, col_desc_str, os.getenv("OPENAI_API_KEY"), verbose=True))
    
    # Generate Sweetviz report (commented out in original)
    #report_file = "sweetviz_report.html"
    #report = sv.analyze(df)
    #report.show_html(report_file)

    # Create a link to open it in new tab
    #st.markdown("### 📊 Open Sweetviz Report")
    #st.markdown(f'<a href="{report_file}" target="_blank">👉 Click here to open report in new tab</a>', unsafe_allow_html=True)
    
    # Store in session state
    st.session_state.df = df
    st.session_state.df_info_str = df_info_str
    st.session_state.col_desc_str = col_desc_str 
    st.session_state.globals_dict = globals_dict
    
    # Set up the LLM
    llm_gr = ChatOpenAI(
        temperature=0.4,
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
    You should consider this conversation to add context to the user query. If user hasn't formed full question, look at the last question to understand what could be his full question.
    
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
        handle_parse_errors=True,
        
    )
    st.session_state.agent = agent
    
    container = st.container()
    callback = StreamlitChatCallbackHandler()
    
    
    # Clean up temporary files (only for uploaded files)
    if data_source == "Upload your own files":
        if desc_path and desc_path.startswith(tempfile.gettempdir()):
            os.unlink(desc_path)
        if csv_path and csv_path.startswith(tempfile.gettempdir()):
            os.unlink(csv_path)
    
    st.success(f"✅ Loaded dataset with shape {df.shape}")

# Display data preview if available
if st.session_state.df is not None:
    with st.expander("Preview Data"):
        st.dataframe(st.session_state.df.head(10))
        
    # Display column descriptions if available
    if st.session_state.col_desc_str:
        with st.expander("High level data observation"):
            st.markdown(st.session_state.col_desc_str)

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
                
                        result = st.session_state.agent.run(input=final_query, callbacks=[callback])
                        
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
    if data_source == "Upload your own files":
        st.info("Please upload a CSV file to start chatting!")
    else:
        st.info("Please select a sample dataset to start chatting!")


# Compact footer with LinkedIn link that matches sidebar background
st.markdown("---")
st.markdown("""
<div class="compact-footer">
    <div class="footer-content">
        <span>Built with <span class="heart">♥️</span> by <span class="name">Monil Shah</span></span>
        <a href="https://www.linkedin.com/in/monil-shah-b9b4911a/" target="_blank" class="linkedin-link">
            <span class="linkedin-icon">in</span>
        </a>
    </div>
</div>

<style>
.compact-footer {
    text-align: center;
    margin-top: 20px;
    padding: 8px;
    border-radius: 8px;
    box-shadow: 0 -1px 4px rgba(0,0,0,0.1), 0 1px 4px rgba(0,0,0,0.1);
}

/* Match sidebar background color in both themes */
html[data-theme="light"] .compact-footer {
    background-color: #f0f2f6;
}

html[data-theme="dark"] .compact-footer {
    background-color: #262730;
}

.footer-content {
    display: flex;
    align-items: center;
    justify-content: center;
    flex-wrap: wrap;
    gap: 8px;
    font-size: 0.85rem;
}

html[data-theme="light"] .footer-content {
    color: #31333F;
}

html[data-theme="dark"] .footer-content {
    color: #FAFAFA;
}

.separator {
    color: #aaa;
}

.heart {
    color: #ff4b4b;
    font-size: 1rem;
    animation: heartbeat 1.5s infinite;
    display: inline-block;
    position: relative;
    top: 1px;
}

.name {
    font-weight: bold;
    background: linear-gradient(90deg, #007bff, #6610f2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.linkedin-link {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    text-decoration: none;
    border-radius: 4px;
    transition: all 0.3s ease;
    background-color: #0077B5;
}

.linkedin-link:hover {
    transform: translateY(-2px);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    opacity: 0.9;
}

.linkedin-icon {
    font-weight: bold;
    color: white;
    font-size: 0.75rem;
}

@keyframes heartbeat {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.2); }
}
</style>
""", unsafe_allow_html=True)
