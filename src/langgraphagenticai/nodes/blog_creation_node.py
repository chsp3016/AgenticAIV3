from src.langgraphagenticai.state.state import State
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AnyMessage
class BlogCreationNode:
    """
    Blog Post creation logic .
    """
    def __init__(self,model):
        self.llm = model


    def create_blogtitle(self):
        """
        Returns a blog title function.
        """


        def blogtitle_node(state: State):
            """
            Blog creation logic for processing the input state and returning a response.
            """
            sys_msg = SystemMessage(content="You are a helpful assistant tasked with the creation of a blog for a given subject")

            return {"messages": [self.llm.invoke([sys_msg] + state["messages"])]}

        return blogtitle_node
    
    
    def create_blogcontent(self):
        """
        Returns a blog content function considering the output.
        """
        def blogcontent_node(state: State):
            """
            Blog content creation logic for processing the input state and returning a response.
            """

            return {"messages": [self.llm.invoke(state["messages"])]}

        return blogcontent_node