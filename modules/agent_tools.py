# modules/agent_tools.py
from langchain.agents import Tool
from langchain_experimental.tools.python.tool import PythonAstREPLTool

class ReturnOutputPythonTool(PythonAstREPLTool):
    """
    An extension of PythonAstREPLTool that returns the output of the last expression.
    """
    def run(self, code: str) -> str:
        try:
            # Evaluate the code and capture output
            local_vars = {}
            exec(code, self.globals, local_vars)
            # Try to get the value of the last expression (last line)
            last_expr = code.strip().split('\n')[-1]
            # Evaluate last expression to get the value
            result = eval(last_expr, self.globals, local_vars)
            return repr(result)
        except Exception as e:
            return f"Error: {e}"

def create_python_tool(globals_dict, col_desc_str=""):
    """
    Create a Python tool for the agent to use.
    
    Args:
        globals_dict (dict): Dictionary of global variables for the Python tool.
        col_desc_str (str, optional): String containing column descriptions.
        
    Returns:
        list: List of Tool objects.
    """
    python_tool = ReturnOutputPythonTool()
    python_tool.globals = globals_dict
    
    return [
        Tool(
            name="python",
            func=python_tool.run,
            description=(
                "When user ask any question, the answer is assumed to be in the 'df' dataframe."
                "Use this to execute Python code. The tool is called 'python'.  — do NOT use square brackets like [python]. Use it as: Action: python"
                " 'df' is already loaded and is available for use. "
                "Do NOT attempt to load files yourself. "
                "You can access and analyze `df` directly."
                "You can use pandas operations like df.head(), df.describe(), "
                "df['column'].value_counts(), etc."
                f"whereever you need to know more details about what each column means, you can refer to {col_desc_str} for more details about the columns. "
            ),
        )
    ]
