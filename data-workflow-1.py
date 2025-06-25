import os
import pandas as pd
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# load the .env file into script
from dotenv import load_dotenv
load_dotenv() # loads variables from .env into os.environ


# this workflow is meant to move some data around in the background to test workflow and MCP configurations
os.environ["OPENAI_API_KEY"] = os.getenv("GROQ_API_KEY") or "your_groq_api_key_here"
os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"

# build the gpt_configuration object
base_llm_config = {
    "config_list": [
        {
            "model": "llama3-8b-8192", # update this model
            "api_key": os.environ["OPENAI_API_KEY"],
            "base_url": os.environ["OPENAI_API_BASE"],
        }
    ],
    "temperature": 0.0,
    "cache_seed": None,
    "timeout": 600,
}

# define the file path
CSV_PATH = "data.csv"
OUTPUT_PATH = "processed_data.csv"
MIDPOINT_PATH = "midpoint_data.csv"

# code execution settings
execution_config = {"use_docker": False, "work_dir": "code"}

# define agent A (read and describe the data, and later save the data)
agent_a = AssistantAgent(
    name="AgentA",
    llm_config=base_llm_config,
    system_message=(
        "You are Agent A. Your role is to:\n"
        "- Step 1: Load and describe the data.\n"
        "- Step 3: Load the transformed data from '{MIDPOINT_PATH}' and save the final result after the transformation.\n"
        "Do not proceed to Step 3 unless Step 2 is confirmed complete.\n"
        "Use this format for code:\n"
        "```python\n# your code here\n```"
    ),
    code_execution_config=execution_config,
)


# define agent B (transforms the data by adding columns or filtering)
agent_b = AssistantAgent(
    name="AgentB",
    llm_config=base_llm_config,
    system_message=(
        "You are Agent B. Your role is to:\n"
        "- Step 2: Receive data description from Agent A and transform it.\n"
        "  * Add a column 'is_adult' (True if age >= 18)\n"
        "  * Filter to include only rows where department contains 'Engineering'\n"
        "  * Save the transformed data to '{MIDPOINT_PATH}' so that Agent A can handle final saving.\n"
        "When writing code, always use triple backticks with 'python'. For example:\n"
        "```python\n# your code here\n```"
    ),
    code_execution_config=execution_config,
)


# define user proxy agent
user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    code_execution_config=execution_config,
)

# set up groupchat and groupchat manager
groupchat = GroupChat(
    agents=[user_proxy, agent_a, agent_b], 
    messages=[],
    max_round=10, # max_round controls the total number of back-and-forth exchanges between agents
    select_speaker_auto_llm_config=base_llm_config,
    )
manager = GroupChatManager(
    groupchat=groupchat, 
    name="MCPWorkflowManager", 
    llm_config=base_llm_config,
    )
# start the workflow
user_proxy.initiate_chat(
    manager,
    message=f"""
    This is a 3-step MCP-based workflow. Follow the instructions exactly.

    Step 1 — Agent A:
    Load and describe the CSV file at '{CSV_PATH}'. Output a summary and data preview.

    Step 2 — Agent B:
    After Agent A's description, transform the data:
    - Add column 'is_adult' (True if age >= 18)
    - Filter to include only rows with department 'Engineering'
    - Save the transformed data to '{MIDPOINT_PATH}'

    Step 3 — Agent A:
    Load '{MIDPOINT_PATH}' and save the final DataFrame to '{OUTPUT_PATH}'.

    All agents: respond only when it's your step.
    """
)
