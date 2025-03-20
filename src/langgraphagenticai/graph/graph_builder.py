from langgraph.graph import StateGraph, START,END, MessagesState
from langgraph.prebuilt import tools_condition,ToolNode
from langchain_core.prompts import ChatPromptTemplate
from src.langgraphagenticai.state.state import State
from src.langgraphagenticai.nodes.basic_chatbot_node import BasicChatbotNode
from src.langgraphagenticai.nodes.chatbot_with_Tool_node import ChatbotWithToolNode
from src.langgraphagenticai.nodes.blog_creation_node import BlogCreationNode
from src.langgraphagenticai.tools.serach_tool import get_tools,create_tool_node
from src.langgraphagenticai.nodes.code_review_node import CodeReviewNode
from src.langgraphagenticai.tools.check_review import check_codereview
import traceback 

class GraphBuilder:

    def __init__(self,model):
        self.llm=model
        self.graph_builder=StateGraph(State)

    def basic_chatbot_build_graph(self):
        """
        Builds a basic chatbot graph using LangGraph.
        This method initializes a chatbot node using the `BasicChatbotNode` class 
        and integrates it into the graph. The chatbot node is set as both the 
        entry and exit point of the graph.
        """
        
        self.basic_chatbot_node=BasicChatbotNode(self.llm)
        self.graph_builder.add_node("chatbot",self.basic_chatbot_node.process)
        self.graph_builder.add_edge(START,"chatbot")
        self.graph_builder.add_edge("chatbot",END)


    def chatbot_with_tools_build_graph(self):
        """
        Builds an advanced chatbot graph with tool integration.
        This method creates a chatbot graph that includes both a chatbot node 
        and a tool node. It defines tools, initializes the chatbot with tool 
        capabilities, and sets up conditional and direct edges between nodes. 
        The chatbot node is set as the entry point.
        """
        ## Define the tool and tool node
        tools=get_tools()
        tool_node=create_tool_node(tools)

        ##Define LLM
        llm = self.llm

        # Define chatbot node
        obj_chatbot_with_node = ChatbotWithToolNode(llm)
        chatbot_node = obj_chatbot_with_node.create_chatbot(tools)

        # Add nodes
        self.graph_builder.add_node("chatbot", chatbot_node)
        self.graph_builder.add_node("tools", tool_node)

        # Define conditional and direct edges
        self.graph_builder.add_edge(START,"chatbot")
        self.graph_builder.add_conditional_edges("chatbot", tools_condition)
        self.graph_builder.add_edge("tools","chatbot")

    
    def blog_creation_build_graph(self):
        """
        Builds an blog creation graph 
        This method creates a title creation node and content cration node. It defines title creator node and content creation node and direct edges between nodes. 
        The title creator node is set as the entry point.
        """
        ##Define LLM
        llm = self.llm

        # Define chatbot node
        obj_blogcreation_with_node = BlogCreationNode(llm)
        blogtitlecreation_node = obj_blogcreation_with_node.create_blogtitle()
        blogcontentcreation_node = obj_blogcreation_with_node.create_blogcontent()

        self.graph_builder.add_node("titleCreator", blogtitlecreation_node)
        self.graph_builder.add_node("contentCreator", blogcontentcreation_node)

        # Define the flow
        self.graph_builder.add_edge(START, "titleCreator")
        self.graph_builder.add_edge("titleCreator", "contentCreator")
        self.graph_builder.add_edge("contentCreator", END)
        
    def code_review_build_graph(self):
        """
        Builds an code creation and review graph 
        This method creates a code creation node, peer review and manager review node. It defines direct and conditional edges between nodes. 
        The code creator node is set as the entry point.
        """
        try:
            
            ##Define LLM
            llm = self.llm

            # Define chatbot node
            obj_codereview_with_node = CodeReviewNode(llm)
            createcode_node = obj_codereview_with_node.create_code()
            createreview_node = obj_codereview_with_node.create_review()
            createmanagerreview_node = obj_codereview_with_node.review_manager()
            
            self.graph_builder.add_node("Create Code", createcode_node)
            self.graph_builder.add_node("Create Peer Review", createreview_node)
            self.graph_builder.add_node("Create Manager Review", createmanagerreview_node)
            
            # Define the flow
            self.graph_builder.add_edge(START, "Create Code")
            self.graph_builder.add_edge("Create Code","Create Peer Review")
            print(f"Before conditional edge Create Peer Review -->Manager review")
            self.graph_builder.add_conditional_edge("Create Peer Review", self.check_codereview)
            self.graph_builder.add_conditional_edge(
            "Create Peer Review",
            self.check_codereview,  # Decision function
            {
                "Pass":  "Create Manager Review","Fail": END               
            }
            )
            print(f"Before conditional edge Manager review-->END")
            self.graph_builder.add_conditional_edge("Create Manager Review", self.check_codereview,
                self.check_codereview,  # Decision function
            {
                "Pass":  END,"Fail": "Create Code"               
            })
            
        except Exception as e:
            # Print the full stack trace
            print("An error occurred while building the Code Review graph:")
            traceback.print_exc()
            # Raise the error with additional context
            raise ValueError(f"An unexpected error occurred while building the Code Review graph: {e}")
    
    def setup_graph(self, usecase: str):
        """
        Sets up the graph for the selected use case.
        """
        if usecase == "Basic Chatbot":
            print(f"Basic Chatbot option")
            self.basic_chatbot_build_graph()

        if usecase == "Chatbot with Tool":
            print(f"Chatbot  tool option")
            self.chatbot_with_tools_build_graph()

        if usecase == "Blog Creation Tool":
            print(f"Blog creation option")
            self.blog_creation_build_graph()
        
        if usecase == "Code Review Tool":
            print(f"code review option")
            self.code_review_build_graph()
        
        return self.graph_builder.compile()
    




    

