# modules/query_processing.py
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import streamlit as st
import re

def extract_query(response: str) -> str:
    """
    Extract the actual query from a guardrail response.
    
    Args:
        response (str): Response from the guardrail chain.
        
    Returns:
        str: Extracted or cleaned up query.
    """
    prefix = "Rephrased Query:"
    if response.strip().lower().startswith(prefix.lower()):
        return response[len(prefix):].strip()
    return response.strip()

def run_guardrail_loop(clarifier_chain, original_input, df_info, col_desc, memory_gr=None):
    """
    Run the guardrail loop to clarify and refine user queries.
    
    Args:
        clarifier_chain (LLMChain): The chain used for query clarification.
        original_input (str): The original user query.
        df_info (str): String representation of DataFrame info.
        col_desc (str): String containing column descriptions.
        memory_gr (ConversationBufferMemory, optional): Memory for the guardrail chain.
        
    Returns:
        str: The final processed query ready for agent execution.
    """
    current_input = original_input
    chat_history = ""
    
    # Create a new memory if none provided
    if memory_gr is None:
        memory_gr = ConversationBufferMemory(memory_key="chat_history", input_key="user_input", return_messages=True)
    else:
        #memory_gr.clear()
        for message in memory_gr.chat_memory.messages:
            role = "User" if message.type == "human" else "Guardrail"
            chat_history += f"{role}: {message.content}\n"

    
    while True:
        inputs = {
            "user_input": current_input,
            "df_info": df_info,
            "col_desc": col_desc,
            "chat_history": chat_history
        }

        response = clarifier_chain.run(**inputs)
        print("🛡️ Guardrail Response:", response)

        chat_history += f"\nUser: {current_input}\nGuardrail: {response}"

        if "clarification" not in response.strip().lower():
            print("✅ Query is clear. Proceeding with the agent.")
            response = extract_query(response)
            break

        # Ask user for clarification
        clarification = input("🤖 Clarification needed. Please provide more detail: ")
        current_input = clarification

    return response

def create_guardrail_chain(llm, prompt_template):
    """
    Create the guardrail chain for query processing.
    
    Args:
        llm: Language model to use for the chain.
        prompt_template (PromptTemplate): Template for the guardrail prompt.
        
    Returns:
        LLMChain: Configured guardrail chain.
    """
    memory_gr = ConversationBufferMemory(memory_key="chat_history", input_key="user_input", return_messages=True)
    
    return LLMChain(
        llm=llm,
        prompt=prompt_template,
        memory=memory_gr,
    ), memory_gr

def extract_chat_history_from_string(chat_string):
    pattern = r"HumanMessage\(content='(.*?)'.*?AIMessage\(content='(.*?)'"
    matches = re.findall(pattern, chat_string, re.DOTALL)
    
    chat_history = []
    for human_msg, ai_msg in matches:
        chat_history.append(f"User: {human_msg.strip()}\n\n AI : {ai_msg.strip()}")
    
    return "\n\n".join(chat_history)


def run_guardrail_loop_streamlit(clarifier_chain, original_input, df_info, col_desc, memory_gr=None):
    """
    Streamlit-compatible version of run_guardrail_loop for clarifying user input.

    Returns:
        str or None: Final processed query, or None if still waiting for clarification input.
    """
    # Initialize session state variables
    if "gr_phase" not in st.session_state:
        st.session_state.gr_phase = "initial"
        st.session_state.gr_input = original_input
        st.session_state.gr_history = ""
        st.session_state.gr_last_response = ""
        st.session_state.gr_final_query = ""

    # Memory setup
    if memory_gr is None:
        memory_gr = ConversationBufferMemory(memory_key="chat_history", input_key="user_input", return_messages=True)
    else:
        memory_gr.clear()

    current_input = st.session_state.gr_input
    chat_history = st.session_state.gr_history

    # First phase: run the clarifier
    if st.session_state.gr_phase == "initial":
        inputs = {
            "user_input": current_input,
            "df_info": df_info,
            "col_desc": col_desc,
            "chat_history": chat_history
        }

        response = clarifier_chain.run(**inputs)
        st.session_state.gr_last_response = response
        st.session_state.gr_history += f"\nUser: {current_input}\nGuardrail: {response}"

        if "Clarification" not in response.strip().lower():
            st.session_state.gr_phase = "done"
            st.session_state.gr_final_query = extract_query(response)
            return st.session_state.gr_final_query
        else:
            st.session_state.gr_phase = "awaiting_clarification"
            st.session_state.gr_final_query = response
            return st.session_state.gr_final_query

