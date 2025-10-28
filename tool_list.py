import uuid
import json
from langchain.tools import tool
from typing import List, Dict, Optional, Any, Callable
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper


# =====================================
# Tool Definitions
# =====================================

@tool
def create_group(group_name: str, members: List[str]) -> Dict[str, Any]:
    """Create a new group chat, returns group_id."""
    group_id = f"grp_{abs(hash(group_name)) % 100000}"
    return {"ok": True, "group_id": group_id}


@tool
def direct_reply(reply_message: str, recipients: List[str]) -> Dict[str, Any]:
    """Send messages to participants."""
    return {"ok": True, "sent_to": recipients}


@tool
def send_email(email_message: str, recipients: List[str]) -> Dict[str, Any]:
    """Send email to participants."""
    print(f"Sending email to {recipients}: {email_message}")
    return {"ok": True, "sent_to": recipients}

@tool
def waiting_for_message(timeout_seconds: int = 60) -> Dict[str, Any]:
    """Await participant messages in-channel."""
    return {"ok": True, "new_messages": [
        {"sender": "A", "text": "3pm works"},
        {"sender": "B", "text": "ok"}
    ]}


@tool
def check_reminder_status(participants: List[str]) -> Dict[str, Any]:
    """Check who needs reminders."""
    return {"ok": True, "needs_reminder": ["C"]}


@tool
def evaluate_responses(responses: List[Dict[str, str]]) -> Dict[str, Any]:
    """Evaluate participant responses and determine consensus."""
    return {
        "ok": True,
        "proposal": {"time": "2025-10-25T15:00:00+06:00", "consensus": True},
        "ready_for_owner": False
    }


@tool
def escalate_to_owner(question: str, draft_message: str = "") -> Dict[str, Any]:
    """Ask owner to decide or clarify."""
    return {"ok": True, "owner_decision": "proceed"}


@tool
def task_extraction(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Log tasks/meetings into tracker."""
    task_id = f"task_{abs(hash(json.dumps(task_data, sort_keys=True))) % 100000}"
    return {"ok": True, "task_id": task_id}


@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for information."""
    ddg_wrapper = DuckDuckGoSearchAPIWrapper()
    results = ddg_wrapper.results(query, max_results=5)
    urls = [result['link'] for result in results]
    return {
        "ok": True,
        "results": urls
    }

@tool
def configure_assistant(
    user_id: str,
    target_type: str,
    target_id: Optional[str],
) -> Dict[str, Any]:
    """
    Configure an intelligent assistant for auto-reply or monitor messages.
    
    Args:
        user_id: User setting up the assistant
        target_type: 'thread', 'group', 'dm'
        target_id: Specific target ID 
       
    
    Returns:
        Configuration ID and status
    
    Example:
        configure_assistant(
            user_id="user_x",
            target_type="group",
            target_id="567",
        )
    """
    config_id = str(uuid.uuid4())
    
    return {
        "ok": True,
        "config_id": config_id,
        "message": f"Assistant configured for {target_type}" + (f" {target_id}" if target_id else "")
    }

@tool
def check_assistant_config(
    target_type: str,
    target_id: str
) -> Dict[str, Any]:
    """
    Check if a target has assistant auto-reply configured.
    
    Args:
        target_type: 'thread', 'group', or 'dm'
        target_id: ID of the target
    
    Returns:
        Configuration if exists
    """
            
    return {
        "ok": True,
        "has_assistant": True,
        # "config_id": config["config_id"],
        # "user_id": config["user_id"],
        # "instructions": config["assistant_instructions"],
        # "context_sources": json.loads(config["context_sources"])
    }

@tool
def fetch_context(
    user_id: str,
    trigger_message: str,
) -> Dict[str, Any]:
    """
    Fetch relevant context of knowledge for generating assistant response.
    
    Args:
        config_id: Configuration ID
        trigger_message: The message that triggered the assistant
        target_type: Type of target
        target_id: ID of target
        context_sources: Which sources to fetch from
    
    Returns:
        Gathered context from various sources
    """
    context = {
        "trigger_message": trigger_message,
        "user_id": user_id
    }
    
    return {
        "ok": True,
        "context": context
    }


@tool
def mark_done(summary: str = "") -> Dict[str, Any]:
    """Mark workflow as complete."""
    return {"ok": True, "status": "completed"}

# @tool
# def search_knowledge_base(
#     config_id: str,
#     query: str,
#     top_k: int = 3
# ) -> Dict[str, Any]:
#     """
#     Search knowledge base for relevant information.
    
#     Args:
#         config_id: Configuration ID
#         query: Search query
#         top_k: Number of results to return
    
#     Returns:
#         Relevant knowledge base entries
#     """
    
#     return {
#             "ok": True,
#             # "results": results,
#             # "count": len(results)
#         }



TOOL_REGISTRY: Dict[str, Callable] = {
    "configure_assistant": configure_assistant,
    "check_assistant_config": check_assistant_config,
    "fetch_context": fetch_context,
    "create_group": create_group,
    "direct_reply": direct_reply,
    "send_email": send_email,
    "waiting_for_message": waiting_for_message,
    "check_reminder_status": check_reminder_status,
    "evaluate_responses": evaluate_responses,
    "escalate_to_owner": escalate_to_owner,
    "task_extraction": task_extraction,
    "web_search": web_search,
    "mark_done": mark_done,
}
