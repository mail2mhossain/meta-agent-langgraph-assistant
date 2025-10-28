from test_meta_agent import execute_meta_agent
from tool_list import TOOL_REGISTRY

request = """
    Analyze this customer feedback data and identify top 3 improvements:

    DATA:
    {
        "total_responses": 500,
        "sentiment": {"positive": 45%, "neutral": 30%, "negative": 25%},
        "top_issues": [
            {"issue": "Performance", "count": 85, "severity": "high"},
            {"issue": "UI complexity", "count": 62, "severity": "medium"},
            {"issue": "Crashes", "count": 58, "severity": "high"},
            {"issue": "Missing features", "count": 47, "severity": "medium"}
        ],
        "feature_requests": [
            {"feature": "Dark mode", "votes": 156},
            {"feature": "Offline mode", "votes": 134},
            {"feature": "Better export", "votes": 98}
        ],
        "nps_score": 42,
        "churn_risk": "medium"
    }

    Prioritize improvements that will:
    1. Reduce churn
    2. Address high-severity issues
    3. Improve NPS score
"""

result = execute_meta_agent(request, TOOL_REGISTRY)

agent_results = result["agent_results"]

agent_name, agent_result = next(reversed(agent_results.items()))
print("\n" + "─"*80)
print(f"Agent: {agent_name}")
print(f"Result: {agent_result}")
print("─"*80)