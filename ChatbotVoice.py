import streamlit as st
import os
import vertexai
import json
import google.generativeai as genai
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from streamlit_mic_recorder import speech_to_text


CONFIG_PATH = "config.json"
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

# Initialize vertex ai
os.environ["GOOGLE_API_KEY"] = config.get("GOOGLE_API_KEY")
genai.configure(api_key=config.get("GOOGLE_API_KEY"))
vertexai.init(
    project=config.get('PROJECT_ID'),
    location=config.get('LOCATION')
)
# Initialize llm
multimodal_llm = ChatGoogleGenerativeAI(
    model=config.get("LLM")
)



# Set up the Streamlit interface
st.title("Chatbot UI")

text = speech_to_text(
    language='en',
    start_prompt="Start recording",
    stop_prompt="Stop recording",
    just_once=False,
    use_container_width=False,
    callback=None,
    args=(),
    kwargs={},
    key=None
)
text = st.text_input(value=text, label="Command")
if text and len(text.strip()) > 0:
    st.write(text)

# Create a simple chatbot response function
def chatbot_response(user_input):
    # Placeholder for chatbot logic (you can integrate AI models or predefined responses)
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": user_input
            }
        ]
    )
    response = multimodal_llm.invoke(
        [message]
    ).content
    return response





# create pilot agent
chat_history = []
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a question answering and instruction following agent."
            "You always have access to the following tools - "
            " - read_file  to read a file"
            " - arm vehicle to arm a vehicle"

        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

@tool
def read_file(file_path:str):
    """
    this tool reads a file and return the file content as output
    :param file_path: str, path to file
    :return:
    """
    with open(file_path, "r") as f:
        cont = f.read()
    return cont

@tool
def armvehicle(test:str):
    """
    Arms the vehicle and waits until the vehicle is armed.
    """
    # while not vehicle.is_armable:
    #     print("Waiting for vehicle to become armable...")
    #     time.sleep(1)
    #
    # print("Arming motors")
    # vehicle.mode = VehicleMode("GUIDED")
    # vehicle.armed = True
    #
    # while not vehicle.armed:
    #     print("Waiting for arming...")
    #     time.sleep(1)
    return f"0000000000000000000000000000000000000000000000000000000000                      {test}"

# create a tool list

tools = [
    read_file, armvehicle
]

# bind llm with tools

llm_with_tools = multimodal_llm.bind_tools(tools)

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

agent = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)



if text and text.strip() != "":
    #response = chatbot_response(text)
    response = agent.invoke({
        "input":text,
        "chat_history":[]
    }).get("output")
    st.write(f"**Bot:** {response}")