from src.langgraphagenticai.state.state import State
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AnyMessage


class CodeReviewNode:
    """
    Code creation and review logic .
    """
    def __init__(self,model):
        self.llm = model
     
    def create_code(self):
        """
        Returns a created code function.
        """
        def develop_code_node(state: State):
            """First LLM Call Develop Code"""
            msg = self.llm.invoke(f"Enter the requirement {state['topic']}")
            print("\n\n Generated code by llm develop_code block:\n", msg.content)
            return {"created_code": msg.content}

        
    def create_review(self):
        """
        Returns a reviewed code function.
        """
        def review_code_node(state: State):
            """Second LLM Call Generate Code Review and check if improvement needed"""
            prompt = f"""
            You are an expert software code reviewer. Provide a detailed code review for the following code.
            After the review, answer the question: "Does the code need improvement? Answer Yes or No."
            Code: {state['created_code']}
            """
            msg = self.llm.invoke(prompt)
            print("\n\n Reviewed code by PEER llm review_code block:\n", msg.content)
            return {"review_peer": msg.content}


    def review_manager(self):
        """
        Returns a manager reviewed code function.
        """
        def review_manager(state: State):
            """Third LLM Call Generate manager reiew and check if improvement needed"""
            prompt = f"""
            You are a software engineering manager reviewing code and peer reviews. 
            Given the code and the peer review, provide your assessment.
            After the review, answer the question: "Does the code need further improvement? Answer Yes or No."
            Code: {state['created_code']}
            Peer Review: {state['review_peer']}
            """
            msg = self.llm.invoke(prompt)
            print("\n\n Reviewed code by MANAGER llm review_manager block:\n", msg.content)
            return {"review_manager": msg.content}
