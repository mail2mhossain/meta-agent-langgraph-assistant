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
from message_classifier import classify_message
from filter_tool_agent import filter_tools_with_agent
from meta_plan_generator import make_plan
from multi_agent_graph_builder import build_multi_agent_graph
from tool_list import TOOL_REGISTRY

llm = get_llm("google")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("meta_agent")


# =====================================
# Main Execution
# =====================================

def execute_meta_agent(
    user_request: str, 
    tool_registry: Dict[str, Callable], 
    max_agents: int = 5,
    skip_simple_chat: bool = True
):
    """Execute the full meta-agent workflow."""
    
    log.info(f"[META] Processing request: {user_request}")
    
    # Step 1: Classify message type
    log.info("[META] Step 1: Classifying message type...")
    try:
        message_type = classify_message(user_request)
        log.info(f"[META] Message type: {message_type.type} (confidence: {message_type.confidence:.2f})")
    except Exception as e:
        log.error(f"[META] Classification failed: {e}")
        message_type = None

    # Step 1.5: Handle simple chat
    if skip_simple_chat and message_type and message_type.type == "simple_chat":
        log.info("[META] Simple chat message - no action required")
        return {
            "ok": True,
            "status": "simple_chat",
            "message_type": message_type.type,
            "confidence": message_type.confidence
        }
    
    # Step 2: Filter tools
    log.info(f"[META] Step 2: Filtering tools from {len(tool_registry)} total...")
    try:
        filtered_tools = filter_tools_with_agent(
            message=user_request,
            all_tools=tool_registry,
            max_tools=20  # Limit tools passed to meta-agent
        )

        filtered_tool_registry: Dict[str, Callable] = {
            name: TOOL_REGISTRY[name]  # name is str, TOOL_REGISTRY[name] is Callable
            for name in filtered_tools  # Iterate over list of strings
            if name in TOOL_REGISTRY  # Check if string exists as key
        }
        
        log.info(f"[META] Filtered: {len(tool_registry)} → {len(filtered_tool_registry)} tools")
        log.info(f"[META] Selected tools: {list(filtered_tool_registry.keys())}")
        
    except Exception as e:
        log.error(f"[META] Filtering failed: {e}, using all tools")
        filtered_tool_registry = tool_registry

    # Safety check: Ensure we have tools
    if not filtered_tool_registry:
        log.warning("[META] No tools selected, falling back to all tools")
        filtered_tool_registry = tool_registry

    # ============================================
    # STAGE 2: PLAN & EXECUTE
    # ============================================
    
    # Step 3: Generate plan using filtered tools
    log.info(f"[META] Step 3: Generating plan with {len(filtered_tool_registry)} tools...")
    try:
        plan = make_plan(llm, user_request, filtered_tool_registry, max_agents)
        log.info(f"[META] Generated plan with {len(plan.agents)} agents")
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
        
    except Exception as e:
        log.error(f"[META] Planning failed: {e}")
        return {
            "ok": False,
            "error": "planning_failed",
            "details": str(e)
        }
    
    # Step 4: Build graph with FILTERED tools (critical!)
    log.info("[META] Step 4: Building multi-agent graph...")
    try:
        graph = build_multi_agent_graph(
            plan, 
            filtered_tool_registry  
        )
        
    except Exception as e:
        log.error(f"[META] Graph building failed: {e}")
        return {
            "ok": False,
            "error": "graph_build_failed",
            "details": str(e)
        }

    # Step 5: Execute
    log.info("[META] Step 5: Starting execution...")
    initial_state = {
        "messages": [],
        "task": user_request,
        "current_agent": plan.entrypoint,
        "iterations": 0,
        "final_ready": False,
        "agent_results": {},
        # Additional metadata
        "filtered_tools": list(filtered_tool_registry.keys()),
        "total_tools_available": len(tool_registry),
        "message_type": message_type.type if message_type else "unknown"
    }

    try:
        final_state = graph.invoke(initial_state)
        
        log.info(f"[META] Execution complete after {final_state.get('iterations', 0)} iterations")
        
        return {
            "ok": True,
            "status": "success",
            "agent_results": final_state.get("agent_results", {}),
            "iterations": final_state.get("iterations", 0),
            "filtered_tools_count": len(filtered_tool_registry),
            "total_tools_count": len(tool_registry),
            "message_type": message_type.type if message_type else "unknown"
        }
        
    except Exception as e:
        log.error(f"[META] Execution failed: {e}")
        return {
            "ok": False,
            "error": "execution_failed",
            "details": str(e)
        }


# =====================================
# Test Examples
# =====================================

def main():
    """Test the LangGraph v1 implementation."""
    
    message = "Schedule a meeting with team members A, B, C for tomorrow at 3pm"
    # message = "User A wants to auto reply messages from user_id 34"
    # message = "User A ask for information about CommChat"
    message = "Research recent quantum computing developments and email me a summary"
    
    result2 = execute_meta_agent(
        message,
        TOOL_REGISTRY
    )
    if result2:
        print(f"\n✅ Task completed successfully!")
        print(f"Iterations: {result2['iterations']}")
        agent_results = result2["agent_results"]

        agent_name, agent_result = next(reversed(agent_results.items()))
        print("\n" + "─"*80)
        print(f"Agent: {agent_name}")
        print(f"Result: {agent_result}")
        print("─"*80)


if __name__ == "__main__":
    main()