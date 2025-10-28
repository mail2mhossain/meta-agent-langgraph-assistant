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
from tool_list import TOOL_REGISTRY
from meta_plan_generator import make_plan
from filter_tool_agent import filter_tools_with_agent

llm = get_llm("google")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("meta_agent")


# =====================================
# Extended State for Multi-Agent Graph
# =====================================

class MultiAgentState(MessagesState):
    """Extended state for multi-agent coordination."""
    task: str
    current_agent: str
    iterations: int
    final_ready: bool
    agent_results: Dict[str, Any]


# =====================================
# Build Multi-Agent Graph using LangGraph v1
# =====================================

def build_multi_agent_graph(plan: Plan, tool_registry: Dict[str, Callable]):
    """Build a multi-agent graph using LangGraph v1's create_react_agent.
    
    Each agent is created using create_react_agent which automatically:
    - Handles the agent-tool calling loop
    - Continues until no more tool calls
    - Manages message history
    """
    
    workflow = StateGraph(MultiAgentState)
    
    # Create agent nodes using create_react_agent
    for spec in plan.agents:
        # Get tools for this agent
        agent_tools = [tool_registry[t] for t in spec.tools_allowed if t in tool_registry]
        
        # Create ReAct agent that handles tool calling automatically
        agent_executor = create_agent(
            name=spec.name,
            model=llm,
            tools=agent_tools,
            system_prompt=spec.full_system_prompt  
        )
        
        # Wrap the agent executor in a state-aware node
        def make_agent_node(agent_name: str, agent_exec):
            def agent_node(state: MultiAgentState) -> MultiAgentState:
                log.info(f"[AGENT] {agent_name} starting")
                
                # Build input for agent
                # The agent gets the task and previous results as context
                context_message = f"Task: {state['task']}"
                if state.get('agent_results'):
                    context_message += f"\n\nPrevious agent results:\n{json.dumps(state['agent_results'], indent=2)}"
                
                # Invoke the agent - it will automatically handle tool calls
                result = agent_exec.invoke({
                    "messages": [HumanMessage(content=context_message)]
                })
                
                # Extract the final message
                final_message = result["messages"][-1]
                
                # Update state
                new_results = state.get("agent_results", {}).copy()
                new_results[agent_name] = final_message.content if hasattr(final_message, 'content') else str(final_message)
                
                log.info(f"[AGENT] {agent_name} completed")
                
                return {
                    **state,
                    "messages": result["messages"],
                    "current_agent": agent_name,
                    "iterations": state.get("iterations", 0) + 1,
                    "agent_results": new_results
                }
            
            return agent_node
        
        workflow.add_node(spec.name, make_agent_node(spec.name, agent_executor))
    
    # Group edges by source to handle multiple conditions from same node
    edges_by_source = {}
    for edge in plan.edges:
        if edge.source not in edges_by_source:
            edges_by_source[edge.source] = []
        edges_by_source[edge.source].append(edge)
    
    # Add edges
    for source, edges in edges_by_source.items():
        # Check if this source has any conditional edges
        conditional_edges = [e for e in edges if e.condition]
        direct_edges = [e for e in edges if not e.condition]
        
        if conditional_edges:
            # Create a routing function that evaluates all conditions
            def make_router(cond_edges: List[EdgeSpec], direct_edges_list: List[EdgeSpec], src: str):
                def route(state: MultiAgentState) -> str:
                    log.debug(f"[ROUTER] Evaluating conditions for {src}")
                    
                    # Try each condition in order
                    for edge in cond_edges:
                        try:
                            result = eval(edge.condition, {"state": state})
                            log.debug(f"[ROUTER] Condition '{edge.condition}' = {result}")
                            if result:
                                log.info(f"[ROUTER] {src} → {edge.target} (condition: {edge.condition})")
                                return edge.target
                        except Exception as e:
                            log.warning(f"[ROUTER] Error evaluating condition '{edge.condition}': {e}")
                            continue
                    
                    # If no conditions match, check if there's a direct edge
                    direct_targets = [e.target for e in direct_edges_list if e.source == src]
                    if direct_targets:
                        log.info(f"[ROUTER] {src} → {direct_targets[0]} (direct edge fallback)")
                        return direct_targets[0]
                    
                    # If we have conditional edges but none matched, go to first target as fallback
                    if cond_edges:
                        fallback_target = cond_edges[0].target
                        log.warning(f"[ROUTER] No conditions matched for {src}, using fallback: {fallback_target}")
                        return fallback_target
                    
                    # Otherwise end
                    log.info(f"[ROUTER] {src} → END (no valid edges)")
                    return END
                return route
            
            # Build path map for all possible targets
            all_targets = set([e.target for e in edges])
            path_map = {target: target for target in all_targets}
            path_map[END] = END
            
            workflow.add_conditional_edges(
                source,
                make_router(conditional_edges, direct_edges, source),
                path_map
            )
        else:
            # Only direct edges from this source
            for edge in direct_edges:
                workflow.add_edge(edge.source, edge.target)
    
    # Set entry point
    workflow.set_entry_point(plan.entrypoint)
    
    # Add stopguard logic for agents without explicit outgoing edges
    if plan.stopguard:
        agents_with_edges = set(e.source for e in plan.edges)
        agents_without_edges = [spec.name for spec in plan.agents if spec.name not in agents_with_edges]
        
        # For agents without edges, add conditional check for stopguard
        for agent_name in agents_without_edges:
            def make_stopguard_check(stop_condition: str):
                def check_stop(state: MultiAgentState) -> str:
                    try:
                        if eval(stop_condition, {"state": state}):
                            log.info(f"[STOPGUARD] Triggered: {stop_condition}")
                            return END
                    except Exception as e:
                        log.warning(f"[STOPGUARD] Error evaluating: {e}")
                    return END
                return check_stop
            
            workflow.add_conditional_edges(
                agent_name,
                make_stopguard_check(plan.stopguard),
                {END: END}
            )
    
    return workflow.compile()


# =====================================
# Main Execution
# =====================================

def execute_meta_agent(user_request: str, tool_registry: Dict[str, Callable], max_agents: int = 5):
    """Execute the full meta-agent workflow."""
    
    log.info(f"[META] Processing request: {user_request}")
    
    tools = filter_tools_with_agent(user_request, TOOL_REGISTRY)
    filtered_tool_registry = {name: tool for name, tool in TOOL_REGISTRY.items() if name in tools}

    # Step 1: Generate plan using structured output
    plan = make_plan(llm, user_request, filtered_tool_registry, max_agents)
    
    log.info(f"[META] Generated plan with {len(plan.agents)} agents")
    
    # Print plan
    print("\n" + "="*60)
    print("MULTI-AGENT EXECUTION PLAN")
    print("="*60)
    
    for i, agent in enumerate(plan.agents, 1):
        print(f"\n{i}. Agent: {agent.name}")
        print(f"   System Prompt: {agent.full_system_prompt}")
        print(f"   Tools: {', '.join(agent.tools_allowed) or 'None (reasoning only)'}")

    
    print(f"\n\nWorkflow:")
    for edge in plan.edges:
        cond = f" (if {edge.condition})" if edge.condition else ""
        print(f"  {edge.source} → {edge.target}{cond}")
    
    print("\n" + "="*60 + "\n")
    
    # Step 2: Build graph with ReAct agents
    try:
        graph = build_multi_agent_graph(plan, tool_registry)
        
        # Step 3: Execute
        initial_state = {
            "messages": [],
            "task": user_request,
            "current_agent": plan.entrypoint,
            "iterations": 0,
            "final_ready": False,
            "agent_results": {}
        }
        
        log.info("[META] Starting execution...")
        final_state = graph.invoke(initial_state)
        
        log.info("[META] Execution complete")
        
        # Print results
        print("\n" + "="*60)
        print("EXECUTION RESULTS")
        print("="*60)
        print(f"Total iterations: {final_state['iterations']}")
        print(f"\nAgent outputs:")
        for agent_name, result in final_state['agent_results'].items():
            print(f"\n{agent_name}:")
            result_str = str(result)
            print(f"  {result_str[:200]}..." if len(result_str) > 200 else f"  {result_str}")
        print("\n" + "="*60)
        
        return final_state
        
    except Exception as e:
        log.error(f"[META] Execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# =====================================
# Test Examples
# =====================================

def main():
    """Test the LangGraph v1 implementation."""
    print("\n" + "="*80)
    print("META-AGENT SYSTEM - LangGraph v1 Implementation")
    print("="*80)
    print("\nUsing create_react_agent with automatic tool calling loop")
    print("Each agent automatically handles tool calls until completion\n")
    
    message = "Schedule a meeting with team members A, B, C for tomorrow at 3pm"
    message = "Research recent developments in quantum computing and summarize key findings"
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