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
#from IPython.display import Image, display

llm = ChatGroq(model="qwen-2.5-32b")

class Route(BaseModel): 
    step: Literal["billing","sales","tech_support","uncertain"] = Field(..., description="Choose response type based on user request")


router = llm.with_structured_output(Route)

class State(TypedDict):
    input:str
    decision:str
    output:str

def billing_response(state:State):
    print(""" This is Billing call 1 for the Route """)
    msg = llm.invoke(state["input"])
    return {"output":msg.content}

def sales_response(state:State):
    print(""" This is sales call for the Route """)
    msg = llm.invoke(state["input"])
    return {"output":msg.content}


def tech_support_response(state:State):
    print(""" This is tech_support for the  Route """)
    msg = llm.invoke(state["input"])
    return {"output":msg.content}

def uncertain_response(state:State):
    print(""" This is uncertain for the  Route """)
    msg = llm.invoke(state["input"])
    return {"output":msg.content}

def email_router_call(state:State):
    decision = router.invoke(
        [SystemMessage(
            content = """Analyze the user's email and classify ONE appropriate response type:
- BILLING for billing requests
- SALES for sales requests 
- Technical Support for tech_support requests  
- uncertain for unsure requests"""
            ),
            HumanMessage(content = state["input"])])
    return {"decision":decision.step}

def email_route_decision(state:State):
    decision = email_router_call(state)
    if state["decision"] == "billing":
        return "billing"
    elif state["decision"] == "sales":
        return "sales"
    elif state["decision"] == "tech_support":
        return "tech_support"
    elif state["decision"] == "uncertain":
        return "uncertain"
    

router_builder =StateGraph(State)
router_builder.add_node("router",email_router_call)
router_builder.add_node("billing",billing_response)
router_builder.add_node("sales",sales_response)
router_builder.add_node("tech_support",tech_support_response)
router_builder.add_node("uncertain",uncertain_response)
router_builder.add_edge(START,"router")
router_builder.add_conditional_edges("router",email_route_decision,)
router_builder.add_edge("billing",END)
router_builder.add_edge("sales",END)
router_builder.add_edge("tech_support",END)
router_builder.add_edge("uncertain",END)

router_workflow = router_builder.compile()

# Show the workflow
#display(Image(workflow.get_graph().draw_mermaid_png()))

# ========== TESTING ==========
test_cases = [
    "Can you calculate the total value of the purchased products?",
    "404 error while loading the website",
    "new customer inquiry",
    "How do I reset my password?",
    "Make me laugh with a programming humor"    
]

for i, test_input in enumerate(test_cases, 1):
    print(f"\n\nTEST CASE {i}: {test_input}")
    try:
        result = router_workflow.invoke({"input": test_input})
        print("\nRESULT:", result["output"])
    except Exception as e:
        print("ERROR:", str(e))
