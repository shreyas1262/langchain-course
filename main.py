import re
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.tools.render import render_text_description
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from typing import Union
from langchain_core.agents import AgentAction, AgentFinish
from callbacks import AgentCallbackHandler

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of the given text by characters"""

    print(f"get_text_length received: {text}")
    text = text.strip("'\n").strip('"')
    return len(text)


def find_tool_by_tool_name(tools, tool_name):

    for tool in tools:
        if tool.name == tool_name:
            return tool

    raise ValueError(f"Tool with name {tool_name} not found")


def main():
    print("Hello ReAct Langchain!")
    tools = [get_text_length]

    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([tool.name for tool in tools]),
    )

    # Here we set stop to "Observation:" so the LLM stops after generating an action and does not try to generate the observation itself
    llm = ChatOpenAI(temperature=0, stop=["Observation:"], callbacks=[AgentCallbackHandler()])
    
    # Build the agent chain
    # The agent takes in the input question and the agent_scratchpad (previous thoughts, actions, observations) and feeds them to the prompt which is then passed to the LLM
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: x.get("agent_scratchpad", ""),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    agent_scratchpad = ""

    # Loop here to simulate the ReAct process
    # The loop is needed because the llm stops after one action/observation cycle
    for i in range(5):
        agent_step = agent.invoke(
            {
                "input": "What is the length of the text 'Dog' in characters?",
                "agent_scratchpad": agent_scratchpad,
            }
        )

        print(f"Step {i+1}:\n{agent_step}\n")

        # Check if done
        if "Final Answer:" in agent_step:
            final = agent_step.split("Final Answer:")[-1].strip()
            print(f"Final Answer: {final}")
            break

        # Extract action and input
        action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", agent_step)
        action_input_match = re.search(r"Action Input:\s*(.+?)(?:\n|$)", agent_step)

        if action_match:
            tool_name = action_match.group(1).strip()
            tool_input = (
                action_input_match.group(1).strip() if action_input_match else ""
            )

            tool_to_use = find_tool_by_tool_name(tools, tool_name)
            observation = tool_to_use.invoke(tool_input)
            print(f"Observation: {observation}\n")

            agent_scratchpad += agent_step + f"\nObservation: {observation}\nThought: "


if __name__ == "__main__":
    main()
