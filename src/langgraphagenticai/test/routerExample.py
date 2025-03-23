import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from typing_extensions import Literal
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display

llm = ChatGroq(model="qwen-2.5-32b")

class Route(BaseModel): 
    step: Literal["joke","poem","story"] = Field(..., description="Choose response type based on user request")


router = llm.with_structured_output(Route)

class State(TypedDict):
    input:str
    decision:str
    output:str

def llm_call_1(state:State):
    print(""" This is LLM call 1 for the  Router joke """)
    msg = llm.invoke(state["input"])
    return {"output":msg.content}

def llm_call_2(state:State):
    print(""" This is LLM call 1 for the  Router Story """)
    msg = llm.invoke(state["input"])
    return {"output":msg.content}


def llm_call_3(state:State):
    print(""" This is LLM call 1 for the  Router poem """)
    msg = llm.invoke(state["input"])
    return {"output":msg.content}

def llm_router_call(state:State):
    decision = router.invoke(
        [SystemMessage(
            content = """Analyze the user's request and select ONE appropriate response type:
- JOKE for humor requests
- POEM for creative writing requests  
- STORY for narrative requests"""
            ),
            HumanMessage(content = state["input"])])
    return {"decision":decision.step}

def route_decision(state:State):
    decision = llm_router_call(state)
    if state["decision"] == "joke":
        return "llm_call_1"
    elif state["decision"] == "story":
        return "llm_call_2"
    elif state["decision"] == "poem":
        return "llm_call_3"
    

router_builder =StateGraph(State)
router_builder.add_node("router",llm_router_call)
router_builder.add_node("llm_call_1",llm_call_1)
router_builder.add_node("llm_call_2",llm_call_2)
router_builder.add_node("llm_call_3",llm_call_3)
router_builder.add_edge(START,"router")
router_builder.add_conditional_edges("router",route_decision,)
router_builder.add_edge("llm_call_1",END)
router_builder.add_edge("llm_call_2",END)
router_builder.add_edge("llm_call_3",END)

router_workflow = router_builder.compile()

# Show the workflow
#display(Image(workflow.get_graph().draw_mermaid_png()))

# ========== TESTING ==========
test_cases = [
    "Write a joke about penguins at a party",
    "Compose a poem about quantum physics",
    "Tell me a story about a robot learning to love",
    "Make me laugh with a programming humor",
    "Create a haiku about autumn leaves"
]

for i, test_input in enumerate(test_cases, 1):
    print(f"\n\nTEST CASE {i}: {test_input}")
    try:
        result = router_workflow.invoke({"input": test_input})
        print("\nRESULT:", result["output"])
    except Exception as e:
        print("ERROR:", str(e))
