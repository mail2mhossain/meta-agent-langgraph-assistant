from typing import Dict, List
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.agents import create_agent, AgentState  
from langchain.agents.structured_output import ToolStrategy
from llm_config import get_llm
from tool_list import TOOL_REGISTRY

llm = get_llm("google")  


class FilterResult(BaseModel):
    """Result from filter agent."""
    selected_tools: List[str] = Field(
        description="List of tool names that are relevant for this request"
    )
    reasoning: str = Field(
        description="Brief explanation of why these tools were selected"
    )
    confidence: float = Field(
        description="Confidence in tool selection (0-1)"
    )


def create_filter_agent_prompt(message: str, all_tools: Dict[str, BaseTool]) -> str:
    """
    Create prompt for filter agent.
    
    Args:
        message: User's message/request
        all_tools: All available tools
    
    Returns:
        Prompt for filter agent
    """
    
    # Create concise tool descriptions
    tool_list = []
    for name, tool in all_tools.items():
        # Truncate long descriptions for filtering
        desc = tool.description[:200] + "..." if len(tool.description) > 200 else tool.description
        tool_list.append(f"- {name}: {desc}")
    
    tools_text = "\n".join(tool_list)
    
    prompt = f"""You are a tool selection agent. Your job is to identify which tools are relevant for a user request.

        USER REQUEST:
        "{message}"

        AVAILABLE TOOLS ({len(all_tools)}):
        {tools_text}

        YOUR TASK:
        Analyze the request and select ONLY the tools that would be useful for fulfilling it.

        GUIDELINES:
        1. Be selective - only choose tools that are clearly relevant
        2. Aim for 5-20 tools (fewer is better if request is simple)
        3. Consider tools that work together (e.g., fetch data + process data + send results)
        4. Don't include tools that are unrelated to the request
        5. If uncertain, include the tool (better to have too many than too few)

        Return:
        - selected_tools: List of tool names
        - reasoning: Brief explanation (1-2 sentences)
        - confidence: How confident you are (0-1)
    """
    
    return prompt

def filter_tools_with_agent(
    message: str,
    all_tools: Dict[str, BaseTool],
    max_tools: int = 20,
    min_tools: int = 3
) -> Dict[str, BaseTool]:
    """
    Stage 1: Use a simple agent to filter tools.
    
    This is the first stage - fast and focused on tool selection.
    
    Args:
        message: User's message/request
        all_tools: Complete tool registry
        max_tools: Maximum tools to return
        min_tools: Minimum tools to ensure (safety net)
    
    Returns:
        Filtered dictionary of relevant tools
    """
    
    print(f"[FILTER AGENT] Analyzing request: '{message[:60]}...'")
    print(f"[FILTER AGENT] Total tools available: {len(all_tools)}")
    
    # Create prompt
    prompt = create_filter_agent_prompt(message, all_tools)

    agent = create_agent(
        model=llm,
        response_format=ToolStrategy(FilterResult),
        system_prompt="You are a tool selection agent. Your job is to identify which tools are relevant for a user request."
    )
    # try:
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})

    filter_results = result["structured_response"]
    # except Exception as e:
    #     print(f"[FILTER AGENT] Error: {e}")
    #     print(f"[FILTER AGENT] Fallback: returning all tools")
    #     return all_tools
    
    print(f"[FILTER AGENT] Selected {len(filter_results.selected_tools)} tools")
    print(f"[FILTER AGENT] Confidence: {filter_results.confidence:.2f}")
    print(f"[FILTER AGENT] Reasoning: {filter_results.reasoning}")
    
    # Build filtered tool dictionary
    filtered_tools = {}
    for tool_name in filter_results.selected_tools[:max_tools]:
        if tool_name in all_tools:
            filtered_tools[tool_name] = all_tools[tool_name]
        else:
            print(f"[FILTER AGENT] Warning: Tool '{tool_name}' not found in registry")
    
    print(f"[FILTER AGENT] Final tool set: {list(filtered_tools.keys())}\n")
    
    return list(filtered_tools.keys())


if __name__ == "__main__":
    message = "Schedule a meeting with team members A, B, C for tomorrow at 3pm"
    message = "User A wants to monitor messages from user_id 34"
    message = "User A: When we are going to release CommChat"
    filtered_tools = filter_tools_with_agent(message, TOOL_REGISTRY)
