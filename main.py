from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

load_dotenv()

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4")
# This will enable the llm to output structured data as per the AgentResponse schema
structured_llm = llm.with_structured_output(AgentResponse)
react_prompt = hub.pull("hwchase17/react")
react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=[
        "input",
        "agent_scratchpad",
        "tools",
        "tool_names",
    ],
).partial(format_instructions="")

# creates a chain that provides the prompt and tools to the llm
agent = create_react_agent(
    llm=llm, tools=tools, prompt=react_prompt_with_format_instructions
)
# An agent that runs with the given tools and LLM
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# Creates a runnable that extracts the "output" key from a dictionary
extract_output = RunnableLambda(lambda x: x["output"])

chain = agent_executor | extract_output | structured_llm


def main():
    result = chain.invoke(
        {
            "input": "Search for 3 job postings for an AI Engineer using langchain in Bangalore on LinkedIn and list their details"
        }
    )
    print(result)


if __name__ == "__main__":
    main()
