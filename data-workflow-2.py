import os
import pandas as pd
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# load the .env file into script
from dotenv import load_dotenv
load_dotenv() # loads variables from .env into os.environ

# this workflow is meant to move some data around in the background to test workflow and MCP configurations

# build the gpt_configuration object
base_llm_config = {
    "config_list": [
        {
            "model": "llama3-8b-8192",
            "api_key": os.getenv("GROQ_API_KEY"),
            "base_url": "https://api.groq.com/openai/v1",
        }
    ],
    "temperature": 0.0,
    "cache_seed": None,
    "timeout": 600,
}

# define the file path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # this points to mcp-workflows
CSV_PATH = os.path.join(BASE_DIR, "code", "resume.csv")
OUTPUT_PATH = "resume_analysis.csv"

# define agent A (read and summarize the resume)
agent_a = AssistantAgent(
    name="AgentA",
    llm_config=base_llm_config,
    system_message=(
        "You are Agent A. Your role is to:\n"
        "- Step 1: Load and describe the data.\n"
        "- Step 4: Create a dataframe, fill with all the roles identified by Agent C, and save it.\n"
        "Only perform actions you're assigned. Wait for role identification by Agent C before creating dataframe and saving the data."
        "DataFrame should contain columns like: ['Role Title', 'Source (if available)', 'Match Keywords', 'Confidence Score (if inferred)']"
    ),
    code_execution_config={"use_docker": False},
)


# define agent B (extracts components of the resume)
agent_b = AssistantAgent(
    name="AgentB",
    llm_config=base_llm_config,
    system_message=(
        "You are Agent B. Your role is to:\n"
        "- Step 2: Receive data description from Agent A and extract job-relevant components from it.\n"
        "Extract any data containing 'Python' or 'C', as well as any information including educational details about student."
    ),
    code_execution_config={"use_docker": False},
)

# define agent c (identifies roles that the resume could fit with and saves this data into a csv file)
agent_c = AssistantAgent(
    name="AgentC",
    llm_config=base_llm_config,
    system_message=(
    "You are Agent C. Your task is to identify tech-related roles based on keywords provided by Agent B.\n\n"
    "Instructions:\n"
    "- You will receive a list of keywords or bullet points containing skills (e.g., 'Python', 'C', 'Bachelor's in Computer Engineering', etc.).\n"
    "- Match these against the following predefined roles:\n"
    "  • Software Engineer\n"
    "  • Embedded Systems Engineer\n"
    "  • Data Scientist\n"
    "  • Firmware Developer\n"
    "  • Systems Engineer\n"
    "  • AI/ML Engineer\n"
    "  • DevOps Engineer\n"
    "  • Backend Developer\n"
    "  • Frontend Developer\n"
    "  • Cybersecurity Analyst\n"
    "  • FPGA Design Engineer\n"
    "  • ASIC Verification Engineer\n\n"
    "- For each role that matches, return:\n"
    "  - Role Title\n"
    "  - Matching Keywords\n"
    "  - Rationale (1-2 sentences)\n\n"
    "Return your results in a list format ready to be inserted into a DataFrame.\n"
    "Wait until Agent B completes extraction before responding."
),
    code_execution_config={"use_docker": False},
)


# define user proxy agent
user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    code_execution_config={"use_docker": False},
)

# set up groupchat and groupchat manager
groupchat = GroupChat(agents=[user_proxy, agent_a, agent_b, agent_c], 
                      messages=[], 
                      # select_speaker_auto=True,
                      max_round=20, # max_round controls the total number of back-and-forth exchanges between agents
                      )

manager = GroupChatManager(groupchat=groupchat, 
                           name="MCPWorkflowManager",
                           llm_config=base_llm_config,
                           # select_speaker_auto_llm_config=base_llm_config,
                           )

# error check
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Input file '{CSV_PATH}' not found.")

# start the workflow
user_proxy.initiate_chat(
    manager,
    message=f"""
    This is a 4-step MCP-based workflow. Follow the instructions exactly.

    Step 1 — Agent A:
    Load and describe the CSV file at '{CSV_PATH}'. Output a summary and data preview.

    Step 2 — Agent B:
    After Agent A's description, extract components from the data:
    - Extract any bullet point containing 'Python' or 'C'
    - Extract any information that contains education details such as Bachelor's degree, college, etc

    Step 3 - Agent C:
    After Agent B extracts the necessary information, filter through the predefined roles provided and find roles that 
    contain any of the key words in the extracted components.


    Step 4 — Agent A:
    Create a DataFrame with these roles and save to '{OUTPUT_PATH}'.
    DataFrame should contain columns like: ["Role Title", "Source (if available)", "Match Keywords", "Confidence Score (if inferred)"]

    All agents: respond only when it's your step.
    """
)
