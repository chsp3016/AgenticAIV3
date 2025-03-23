from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, List
from langchain_core.messages import HumanMessage, AIMessage

class State(TypedDict,total=False):
    """ 
    Represents the structure of the state used in the graph.
    
    messages: Annotated[list, add_messages] """
    messages: Annotated[list, add_messages] 
    topic: str
    created_code: str
    review_peer: str
    review_manager: str

""" class CodeState(TypedDict):
    requirement: str
    created_code: str
    review_peer: str
    review_manager: str """