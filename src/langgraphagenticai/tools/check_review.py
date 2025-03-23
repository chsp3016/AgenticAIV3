# In src.langgraphagenticai.tools.check_review
from src.langgraphagenticai.state.state import State

# In src.langgraphagenticai.tools.check_review.py

def check_codereview(state: State):
    """Check code review results to determine pass/fail"""
    # Get current review content based on node
    if "review_peer" in state:
        review_text = state["review_peer"]
    elif "review_manager" in state:
        review_text = state["review_manager"]
    else:
        print("No review found in state")
        return "Pass"  # Default to Pass if no review found
    
    print(f"code review called")
    # Only fail if there are major issues, not just minor improvements
    if "Yes" in review_text and ("major issues" in review_text.lower() or "critical" in review_text.lower()):
        print(f"Fail returned")
        return "Fail"
    return "Pass"