import re
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from callbacks import AgentCallbackHandler

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of the given text by characters"""
    print(f"get_text_length received: {text}")
    text = text.strip("'\n").strip('"')
    return len(text)


def main():
    print("Hello ReAct Langchain with bind_tools!")
    tools = [get_text_length]

    llm = ChatOpenAI(temperature=0, callbacks=[AgentCallbackHandler()])
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        HumanMessage(content="What is the length of the text 'Dog' in characters?")
    ]

    for i in range(5):
        print(f"\nStep {i+1}:")
        
        # Invoke LLM with tools
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        print(f"Response: {response.content}")
        
        # Check if LLM wants to use tools
        if not response.tool_calls:
            # No tool calls, means final answer
            print(f"\n✅ Final Answer: {response.content}")
            break
        
        # Execute each tool call
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_input = tool_call["args"]
            
            print(f"Tool: {tool_name}")
            print(f"Input: {tool_input}")
            
            # Find and execute tool
            selected_tool = next((t for t in tools if t.name == tool_name), None)
            if selected_tool:
                observation = selected_tool.invoke(tool_input)
                print(f"Observation: {observation}\n")
                
                # Add tool message to conversation
                messages.append(
                    ToolMessage(
                        content=str(observation),
                        tool_call_id=tool_call["id"]
                    )
                )


if __name__ == "__main__":
    main()