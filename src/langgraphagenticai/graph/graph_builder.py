from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from src.langgraphagenticai.state.state import State
from src.langgraphagenticai.nodes.basic_chatbot_node import BasicChatbotNode
from src.langgraphagenticai.nodes.chatbot_with_Tool_node import ChatbotWithToolNode
from src.langgraphagenticai.nodes.blog_creation_node import BlogCreationNode
from src.langgraphagenticai.tools.serach_tool import get_tools, create_tool_node
from src.langgraphagenticai.nodes.code_review_node import CodeReviewNode
from src.langgraphagenticai.tools.check_review import check_codereview
import traceback 

class GraphBuilder:

    def __init__(self, model):
        self.llm = model
        self.graph_builder = StateGraph(State)

    def basic_chatbot_build_graph(self):
        """
        Builds a basic chatbot graph using LangGraph.
        This method initializes a chatbot node using the `BasicChatbotNode` class 
        and integrates it into the graph. The chatbot node is set as both the 
        entry and exit point of the graph.
        """
        
        self.basic_chatbot_node = BasicChatbotNode(self.llm)
        self.graph_builder.add_node("chatbot", self.basic_chatbot_node.process)
        self.graph_builder.add_edge(START, "chatbot")
        self.graph_builder.add_edge("chatbot", END)


    def chatbot_with_tools_build_graph(self):
        """
        Builds an advanced chatbot graph with tool integration.
        This method creates a chatbot graph that includes both a chatbot node 
        and a tool node. It defines tools, initializes the chatbot with tool 
        capabilities, and sets up conditional and direct edges between nodes. 
        The chatbot node is set as the entry point.
        """
        ## Define the tool and tool node
        tools = get_tools()
        tool_node = create_tool_node(tools)

        ##Define LLM
        llm = self.llm

        # Define chatbot node
        obj_chatbot_with_node = ChatbotWithToolNode(llm)
        chatbot_node = obj_chatbot_with_node.create_chatbot(tools)

        # Add nodes
        self.graph_builder.add_node("chatbot", chatbot_node)
        self.graph_builder.add_node("tools", tool_node)

        # Define conditional and direct edges
        self.graph_builder.add_edge(START, "chatbot")
        self.graph_builder.add_conditional_edges("chatbot", tools_condition)
        self.graph_builder.add_edge("tools", "chatbot")

    
    def blog_creation_build_graph(self):
        """
        Builds an blog creation graph 
        This method creates a title creation node and content creation node. It defines title creator node and content creation node and direct edges between nodes. 
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
        Builds a code creation and review graph.
        This method creates a code creation node, peer review node, and manager review node.
        It defines direct and conditional edges between nodes.
        The code creator node is set as the entry point.
        """
        try:
            ## Define LLM
            llm = self.llm

            print("Debug: Creating CodeReviewNode")
            obj_codereview_with_node = CodeReviewNode(llm)

            # Create wrapper functions that ensure AI messages are generated
            def create_code_wrapper(state):
                print(f"Debug: create_code_wrapper called with state: {state}")
                try:
                    # Call the original create_code function
                    result = obj_codereview_with_node.create_code()(state)
                    print(f"Debug: create_code result: {result}")
                    
                    # Ensure messages list exists
                    if "messages" not in result:
                        result["messages"] = state.get("messages", [])
                    
                    # If there's created_code but no AI message showing it, add one
                    if "created_code" in result and result["created_code"]:
                        # Check if we already have an AI message for this code
                        has_code_message = any(
                            isinstance(msg, AIMessage) and msg.content == result["created_code"]
                            for msg in result["messages"]
                        )
                        
                        if not has_code_message:
                            print("Debug: Adding AIMessage for created_code")
                            result["messages"].append(AIMessage(content=result["created_code"]))
                    
                    return result
                except Exception as e:
                    print(f"Error in create_code_wrapper: {e}")
                    traceback.print_exc()
                    # Return original state with error message
                    return {
                        **state,
                        "messages": state.get("messages", []) + [
                            AIMessage(content=f"Error generating code: {str(e)}")
                        ]
                    }

            def create_review_wrapper(state):
                print(f"Debug: create_review_wrapper called with state keys: {state.keys()}")
                try:
                    # Call the original create_review function
                    result = obj_codereview_with_node.create_review()(state)
                    print(f"Debug: create_review result keys: {result.keys()}")
                    
                    # Ensure messages list exists
                    if "messages" not in result:
                        result["messages"] = state.get("messages", [])
                    
                    # If there's review_peer but no AI message showing it, add one
                    if "review_peer" in result and result["review_peer"]:
                        # Check if we already have an AI message for this review
                        has_review_message = any(
                            isinstance(msg, AIMessage) and msg.content == result["review_peer"]
                            for msg in result["messages"]
                        )
                        
                        if not has_review_message:
                            print("Debug: Adding AIMessage for review_peer")
                            result["messages"].append(AIMessage(content=result["review_peer"]))
                    
                    return result
                except Exception as e:
                    print(f"Error in create_review_wrapper: {e}")
                    traceback.print_exc()
                    # Return original state with error message
                    return {
                        **state,
                        "messages": state.get("messages", []) + [
                            AIMessage(content=f"Error generating peer review: {str(e)}")
                        ]
                    }

            def create_manager_review_wrapper(state):
                print(f"Debug: create_manager_review_wrapper called with state keys: {state.keys()}")
                try:
                    # Call the original manager review function
                    result = obj_codereview_with_node.review_manager()(state)
                    print(f"Debug: manager review result keys: {result.keys()}")
                    
                    # Ensure messages list exists
                    if "messages" not in result:
                        result["messages"] = state.get("messages", [])
                    
                    # If there's review_manager but no AI message showing it, add one
                    if "review_manager" in result and result["review_manager"]:
                        # Check if we already have an AI message for this manager review
                        has_manager_review_message = any(
                            isinstance(msg, AIMessage) and msg.content == result["review_manager"]
                            for msg in result["messages"]
                        )
                        
                        if not has_manager_review_message:
                            print("Debug: Adding AIMessage for review_manager")
                            result["messages"].append(AIMessage(content=result["review_manager"]))
                    
                    return result
                except Exception as e:
                    print(f"Error in create_manager_review_wrapper: {e}")
                    traceback.print_exc()
                    # Return original state with error message
                    return {
                        **state,
                        "messages": state.get("messages", []) + [
                            AIMessage(content=f"Error generating manager review: {str(e)}")
                        ]
                    }

            # Use wrapper functions instead of direct node references
            print("Debug: Adding Create Code node to the graph")
            self.graph_builder.add_node("Create Code", create_code_wrapper)
            
            print("Debug: Adding other nodes to the graph")
            self.graph_builder.add_node("Create Peer Review", create_review_wrapper)
            self.graph_builder.add_node("Create Manager Review", create_manager_review_wrapper)

            print("Debug: Defining edges between nodes")
            # Define the flow
            self.graph_builder.add_edge(START, "Create Code")
            self.graph_builder.add_edge("Create Code", "Create Peer Review")
            self.graph_builder.add_conditional_edges(
                "Create Peer Review",
                check_codereview,  # Decision function
                {
                    "Pass": "Create Manager Review",
                    "Fail": END
                }
            )
            self.graph_builder.add_conditional_edges(
                "Create Manager Review",
                check_codereview,  # Decision function
                {
                    "Pass": END,
                    "Fail": "Create Code"
                }
            )
            print("Debug: Code Review Graph successfully built")

        except RuntimeError as e:
            print("RuntimeError occurred while adding a node to the graph:")
            traceback.print_exc()
            raise ValueError("Failed to add a node to the graph") from e
        except Exception as e:
            print("An error occurred while building the Code Review graph:")
            traceback.print_exc()
            raise ValueError(f"An unexpected error occurred while building the Code Review graph: {e}")
    
    def setup_graph(self, usecase: str):
        """
        Sets up the graph for the selected use case.
        """
        if usecase == "Basic Chatbot":
            print(f"Basic Chatbot option")
            self.basic_chatbot_build_graph()

        if usecase == "Chatbot with Tool":
            print(f"Chatbot tool option")
            self.chatbot_with_tools_build_graph()

        if usecase == "Blog Creation Tool":
            print(f"Blog creation option")
            self.blog_creation_build_graph()
        
        if usecase == "Code Review Tool":
            print(f"Code review option")
            self.code_review_build_graph()
        
        return self.graph_builder.compile()