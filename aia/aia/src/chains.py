from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.utilities.zapier import ZapierNLAWrapper
from .constants import TOOLS, FORMAT_INSTRUCTIONS
from dotenv import load_dotenv
from io import StringIO
import sys
import re
import os

load_dotenv()

USE_GPT4_DEFAULT = False


def get_agent(llm):
    os.environ['SERPAPI_API_KEY'] = os.environ.get('SERPAPI_API_KEY')
    agent_tools = load_tools(TOOLS, llm=llm)
    zapier = ZapierNLAWrapper()
    zapier_toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
    tools = agent_tools + zapier_toolkit.get_tools()
    memory = ConversationBufferMemory(memory_key="chat_history")
    agent = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True,
                             agent_kwargs={
                                 "format_instructions": FORMAT_INSTRUCTIONS},
                             memory=memory,
                             handle_parsing_errors=True)
    return agent


def run_agent(agent, inp, capture_hidden_text):
    hidden_text = None
    os.environ["OPENAI_API_KEY"] = os.environ.get('OPENAI_API_KEY')
    os.environ["ZAPIER_NLA_API_KEY"] = os.environ.get("ZAPIER_NLA_API_KEY")

    if capture_hidden_text:
        tmp = sys.stdout
        hidden_text_io = StringIO()
        sys.stdout = hidden_text_io
        output = agent.run(input=inp)
        sys.stdout = tmp
        hidden_text = hidden_text_io.getvalue()

        # remove escape characters from hidden_text
        hidden_text = re.sub(r'\x1b[^m]*m', '', hidden_text)
        hidden_text = re.sub(
            r"Entering new AgentExecutor chain...\n", "", hidden_text)
        hidden_text = re.sub(r"Finished chain.", "", hidden_text)
        hidden_text = re.sub(r"Thought:", "\n\nThought:", hidden_text)
        hidden_text = re.sub(r"Action:", "\n\nAction:", hidden_text)
        hidden_text = re.sub(r"Observation:", "\n\nObservation:", hidden_text)
        hidden_text = re.sub(r"Input:", "\n\nInput:", hidden_text)
        hidden_text = re.sub(r"AI:", "\n\nAI:", hidden_text)
    else:
        output = agent.run(input=inp)
    return output, hidden_text
