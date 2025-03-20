from src.langgraphagenticai.state.state import State



def __init__(self,model):
    self.llm = model
def check_codereview(self,state: State):
    """Check code similarity with requirement using LLM"""
    prompt = f"""
    You are an expert software code reviewer. Your job is to check if the code meets the requirement and provide suggestions for improvement.
    Requirement: {state['requirement']}
    Code: {state['created_code']} 

    1. Does the code need improvement? Answer "Yes" or "No".
    2. Provide very brief suggestions for improving the code. If no improvements are necessary, answer "None".
    """
    review_result = self.llm.invoke(prompt)

    review_text = review_result.content
    print(f"code review called")
    if "Yes" in review_text :
        print(f"Fail returned")
        return "Fail"  # Fails if code doesn't satisfy the requirement and has suggestions
    return "Pass" #passes if code satisfy the requirement
