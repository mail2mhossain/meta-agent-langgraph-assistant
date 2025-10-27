"""
Meta-Agent Generator using LangGraph v1 API
Uses create_agent with automatic tool calling loop and structured output
"""

from __future__ import annotations
from typing import List, Dict, Optional, Any, Callable
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent, AgentState  
from langchain.tools import tool
from langchain.agents.structured_output import ToolStrategy
from langgraph.graph import StateGraph, END, MessagesState
import json
import uuid
import logging
from llm_config import get_llm
from filter_tool_agent import filter_tools_with_agent
from tool_list import TOOL_REGISTRY

llm = get_llm("google")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("meta_agent")




# =====================================
# Pydantic Schemas
# =====================================


class AgentSpec(BaseModel):
    """Specification for a single agent in the multi-agent system."""
    name: str = Field(description="Unique agent identifier (e.g., 'researcher', 'coordinator')")
    role: str = Field(description="Human-readable purpose (e.g., 'Resolve participant identities')")
    system_prompt: str = Field(description="Detailed system prompt for the agent")
    goal: Optional[str] = Field(
        default=None,
        description="High-level objective shown to the LLM"
    )
    success_criteria: Optional[str] = Field(
        default=None,
        description="Programmatic checks for orchestrator (not shown to LLM)"
    )
    tools_allowed: List[str] = Field(default=[], description="List of tool names this agent can use")
    max_iterations: int = Field(default=3, description="Maximum attempts before failing")

    @property
    def full_system_prompt(self) -> str:
        """Complete system prompt for create_agent().
        
        Format:
            Role: {role}
            Goal: {goal}  (if provided)
            
            {system_prompt}
        """
        parts = [f"Role: {self.role}"]
        
        if self.goal:
            parts.append(f"Goal: {self.goal}")
        
        parts.append("")  # blank line before instructions
        parts.append(self.system_prompt)
        
        return "\n".join(parts)


class EdgeSpec(BaseModel):
    """Specification for a graph edge between agents."""
    source: str = Field(description="Source agent name")
    target: str = Field(description="Target agent name")
    condition: Optional[str] = Field(default=None, description="Condition for edge traversal")


class Plan(BaseModel):
    """Complete multi-agent system plan."""
    agents: List[AgentSpec] = Field(description="List of agent specifications")
    edges: List[EdgeSpec] = Field(description="Connections between agents")
    entrypoint: str = Field(description="Name of the starting agent")
    stopguard: Optional[str] = Field(
        default="state.get('iterations', 0)>=8 or state.get('final_ready', False)==True",
        description="Global stopping condition"
    )
    justification: Optional[str] = Field(
        default=None,
        description="Explanation if more than recommended agents are used"
    )



# =====================================
# Meta-Planner using Structured Output
# =====================================

PLANNER_SYSTEM = """You are an expert multi-agent system architect.

Your task is to design an optimal multi-agent workflow for any given request.

CORE PRINCIPLES:
1. **Analyze the Task**: Deeply understand what needs to be accomplished
2. **Decompose Intelligently**: Break down into logical, sequential or parallel steps
3. **Design Specialized Agents**: Create agents with single, clear responsibilities
4. **Optimize Flow**: Design the most efficient path from input to completion
5. **Use Appropriate Tools**: Assign only the tools each agent needs for its specific role

DESIGN GUIDELINES:
- Agent count: Aim for 2-4 agents; only exceed if genuinely beneficial
- Agent naming: Use domain-appropriate names that reflect actual roles
- Agent instructions: Write clear, specific instructions for each agent's role
- Tool assignment: Give each agent ONLY the tools it needs
- Flow structure: Can be linear, parallel, conditional, or iterative as needed
- Termination: Ensure at least one agent can call mark_done to signal completion
- **Edge conditions**: If using conditional edges, conditions MUST be valid Python expressions
  that evaluate to True/False. Use `state.get('key', default)` to access state variables.
  Examples: 
    * "state.get('consensus', False) == True"
    * "state.get('conflicts_found', False) == True"
    * "state.get('retries', 0) < 3"
  NEVER use plain English like "if consensus" - use proper Python syntax!

FLEXIBILITY:
- For research tasks: Design appropriate discovery and synthesis flow
- For automation tasks: Design appropriate execution and verification flow
- For coordination tasks: Design appropriate communication and tracking flow
- For analysis tasks: Design appropriate data processing and reporting flow
- For creative tasks: Design appropriate generation and refinement flow
- For ANY other domain: Invent the most logical agent structure

CONSTRAINTS:
- Use ONLY the tools provided in the registry
- If no suitable tools exist, design reasoning-only agents
- Ensure clear handoffs between agents
- Avoid unnecessary complexity

OUTPUT:
Return a JSON object matching the Plan schema with thoughtfully designed agents and workflow."""


def make_plan(llm, user_request: str, tool_registry: Dict[str, Callable], max_agents: int = 5) -> Plan:
    """Generate a multi-agent plan using structured output.
    
    This uses LangChain's structured output feature to ensure the LLM
    returns a properly formatted Plan object.
    """
    # Build tool descriptions
    tools_block = []
    for name, tool_func in tool_registry.items():
        desc = tool_func.description if hasattr(tool_func, 'description') else "No description"
        tools_block.append(f"- **{name}**: {desc}")
    
    tools_text = "\n".join(tools_block)
    
    PLANNER_HUMAN = (
        "USER REQUEST:\n"
        "{user_request}\n\n"

        "AVAILABLE TOOLS (name: description)\n"
        "Only these tools are available — do not invent or assume additional ones:\n"
        "{tools_text}\n\n"

        "CONSTRAINTS:\n"
        "- if user requests is a question: Asking for information or clarification, then generate workflow for assistant auto-reply, if the target has assistant auto-reply configured\n"
        "- CommChat, Kindermat is our product. So, to get any info need to call fetch_context tool\n"
        "- Determine the number of agents required based on the task complexity.\n"
        "- HARD CAP: Do not exceed max_agents = 5.\n"
        "- If you propose more than 5 agents, include a 'justification' field "
        "that clearly explains the measurable benefit (e.g., parallelism, compliance, waiting, heterogeneous skills).\n"
        "- Respect all resource limits when defined:\n"
        "- Use only the listed tools; if no suitable tool is available, create a reasoning-only plan.\n"
        "- Define clear handoffs between agents; avoid cyclic dependencies unless strictly necessary.\n"
        "- Stop the process when either 'done' is called or when state.final_ready == true.\n"
    )

    agent = create_agent(
        model=llm,
        response_format=ToolStrategy(Plan),
        system_prompt=PLANNER_SYSTEM

    )

    try:
        result = agent.invoke({"messages": [{"role": "user", "content": PLANNER_HUMAN.format(user_request=user_request, tools_text=tools_text)}]})

        plan = result["structured_response"]
        
        # Validate tool references
        for agent in plan.agents:
            invalid_tools = [t for t in agent.tools_allowed if t not in tool_registry]
            if invalid_tools:
                log.warning(f"Agent '{agent.name}' references unknown tools: {invalid_tools}")
                agent.tools_allowed = [t for t in agent.tools_allowed if t in tool_registry]
        
        # Validate edge conditions
        for edge in plan.edges:
            if edge.condition:
                # Check if condition looks like it might be invalid Python
                if not any(op in edge.condition for op in ['==', '!=', '>', '<', 'state.get', 'state[']):
                    log.warning(
                        f"Edge condition may be invalid Python: '{edge.condition}'. "
                        f"Conditions should use state.get() syntax. "
                        f"This edge will be converted to a direct edge."
                    )
                    # Convert to direct edge
                    edge.condition = None
        
        log.info(f"Generated plan with {len(plan.agents)} agents: {[a.name for a in plan.agents]}")
        
        return plan
        
    except Exception as e:
        import traceback
        log.error(f"Error generating plan: {e}")
        log.debug(f"Full traceback: {traceback.format_exc()}")
        
        # Fallback plan
        log.info("Using fallback single-agent plan")
        return Plan(
            agents=[
                AgentSpec(
                    name="executor",
                    goal="Execute the user request",
                    instructions="You are a general-purpose agent. Complete the user's request to the best of your ability.",
                    tools_allowed=list(tool_registry.keys())[:3],
                )
            ],
            edges=[],
            entrypoint="executor",
            stopguard="state.get('iterations', 0)>=3"
        )


# =====================================
# Test Examples
# =====================================

def main():
    """Test the LangGraph v1 implementation."""
    

    message = "Schedule a meeting with team members A, B, C for tomorrow at 3pm"
    # message = "User A wants to auto reply messages from user_id 34"
    # message = "User A ask for information about CommChat"
    message = "Research recent quantum computing developments and email me a summary"

    tools = filter_tools_with_agent(message, TOOL_REGISTRY)
    filtered_tool_registry = {name: tool for name, tool in TOOL_REGISTRY.items() if name in tools}
    
    plan = make_plan(
        llm,
        message,
        filtered_tool_registry
    )

    print("\n" + "─"*80)
    print(f"Plan:\n{plan.model_dump_json(indent=2)}")
    print("─"*80)

    for agent in plan.agents:
        print("\n" + "─"*80)
        print(f"Agent: {agent.name}")
        print(f"System Prompt: {agent.full_system_prompt}")
        print(f"Tools: {agent.tools_allowed}")
        print(f"Success Criteria: {agent.success_criteria}")
        print("─"*80)

    for edge in plan.edges:
        print("\n" + "─"*80)
        print(f"Edge: {edge.source} -> {edge.target}")
        print(f"Condition: {edge.condition}")
        print("─"*80)

    print("\n" + "─"*80)
    print(f"Entry Point: {plan.entrypoint}")
    print(f"Stop Guard: {plan.stopguard}")
    print("─"*80)


if __name__ == "__main__":
    main()