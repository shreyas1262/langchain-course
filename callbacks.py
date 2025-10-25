
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from typing import Any, Dict, List

class AgentCallbackHandler(BaseCallbackHandler):
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Run when LLM starts running."""
        print(f"***Prompt to LLM was:***\n{prompts[0]}")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        print(f"***LLM response was:***\n{response.generations[0][0].text}")
        print("*******************************")