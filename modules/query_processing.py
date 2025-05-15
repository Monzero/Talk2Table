# modules/query_processing.py
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

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
        memory_gr.clear()
    
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
        current_input = "Original prompt was :" + current_input + " then user clarified :" + clarification

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
