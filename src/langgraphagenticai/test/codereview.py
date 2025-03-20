from langgraph.graph import StateGraph, START, END
#from IPython.display import Image, display
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from typing_extensions import TypedDict
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="qwen-2.5-32b")

class State(TypedDict):
    requirement: str
    created_code: str
    review_peer: str
    review_manager: str

def develop_code(state: State):
    """First LLM Call Develop Code"""
    msg = llm.invoke(f"Enter the requirement {state['requirement']}")
    print("\n\n Generated code by llm develop_code block:\n", msg.content)
    return {"created_code": msg.content}

def check_codeReview(state: State):
    """Check code similarity with requirement using LLM"""
    prompt = f"""
    You are an expert software code reviewer. Your job is to check if the code meets the requirement and provide suggestions for improvement.
    Requirement: {state['requirement']}
    Code: {state['created_code']} 

    1. Does the code need improvement? Answer "Yes" or "No".
    2. Provide very brief suggestions for improving the code. If no improvements are necessary, answer "None".
    """
    review_result = llm.invoke(prompt)
   
    review_text = review_result.content

    if "Yes" in review_text :
        return "Fail"  # Fails if code doesn't satisfy the requirement and has suggestions
    return "Pass" #passes if code satisfy the requirement


def review_code(state: State):
    """Second LLM Call Generate Code Review and check if improvement needed"""
    prompt = f"""
    You are an expert software code reviewer. Provide a detailed code review for the following code.
    After the review, answer the question: "Does the code need improvement? Answer Yes or No."
    Code: {state['created_code']}
    """
    msg = llm.invoke(prompt)
    print("\n\n Reviewed code by PEER llm review_code block:\n", msg.content)
    return {"review_peer": msg.content}


def review_manager(state: State):
    """Third LLM Call Generate manager reiew and check if improvement needed"""
    prompt = f"""
    You are a software engineering manager reviewing code and peer reviews. 
    Given the code and the peer review, provide your assessment.
    After the review, answer the question: "Does the code need further improvement? Answer Yes or No."
    Code: {state['created_code']}
    Peer Review: {state['review_peer']}
    """
    msg = llm.invoke(prompt)
    print("\n\n Reviewed code by MANAGER llm review_manager block:\n", msg.content)
    return {"review_manager": msg.content}

workflow = StateGraph(State)

workflow.add_node("Develop code", develop_code)
workflow.add_node("Peer review", review_code)
workflow.add_node("Manager Review", review_manager)

workflow.add_edge(START, "Develop code")
workflow.add_edge("Develop code", "Peer review")
workflow.add_conditional_edges(
    "Peer review",
    check_codeReview,
    {"Fail": "Develop code", "Pass": "Manager Review"},
)
workflow.add_conditional_edges(
    "Manager Review",
    check_codeReview,
    {"Fail": "Develop code", "Pass": END},
)

chain = workflow.compile()
#display(Image(chain.get_graph().draw_mermaid_png()))
# Example Usage

# Test case 1: Code should pass all reviews
initial_state_pass = {"requirement": "Write a python function to add two numbers"}
result_pass = chain.invoke(initial_state_pass)
print("Test Case 1 (Pass):\n", result_pass)
