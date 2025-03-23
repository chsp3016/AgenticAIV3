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
        return develop_code_node
        
    def create_review(self):
        """
        Returns a reviewed code function.
        """
        def review_code_node(state: State):
            """Second LLM Call Generate Code Review and check if improvement needed"""
            prompt = f"""
            You are an expert software code reviewer. Your job is to check if the code meets the requirement and provide suggestions for improvement.
            Requirement: {state['topic']}
            Code: {state['created_code']} 

            1. Does the code have CRITICAL FLAWS or MAJOR ISSUES that make it unsuitable for use? Answer "Yes" or "No".
            2. Provide a brief assessment of the code quality.
            3. Provide suggestions for improving the code. If no improvements are necessary, answer "None".

            Be clear in your response whether any issues are MAJOR (requiring a complete rewrite) or MINOR (suggestions for improvement).
            """
            msg = self.llm.invoke(prompt)
            print("\n\n Reviewed code by PEER llm review_code block:\n", msg.content)
            return {"review_peer": msg.content}
        return review_code_node

    def review_manager(self):
        """
        Returns a manager reviewed code function.
        """
        def review_manager_node(state: State):
            """Third LLM Call Generate manager reiew and check if improvement needed"""
            prompt = f"""
            You are an expert software code reviewer. Your job is to check if the code meets the requirement and provide suggestions for improvement.
            Requirement: {state['topic']}
            Code: {state['created_code']} 

            1. Does the code have CRITICAL FLAWS or MAJOR ISSUES that make it unsuitable for use? Answer "Yes" or "No".
            2. Provide a brief assessment of the code quality.
            3. Provide suggestions for improving the code. If no improvements are necessary, answer "None".

            Be clear in your response whether any issues are MAJOR (requiring a complete rewrite) or MINOR (suggestions for improvement).
            """
            msg = self.llm.invoke(prompt)
            print("\n\n Reviewed code by MANAGER llm review_manager block:\n", msg.content)
            return {"review_manager": msg.content}
        return review_manager_node