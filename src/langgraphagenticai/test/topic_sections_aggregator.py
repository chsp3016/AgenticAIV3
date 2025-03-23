import os
from dotenv import load_dotenv
load_dotenv()
from typing_extensions import Annotated
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from typing_extensions import Literal
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send

from typing import Annotated, List
import operator

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
#os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")


class Section(BaseModel):
    number: int = Field(..., description="Number of the section")
    name: str = Field(..., description="Name of the section")
    description : str = Field(..., description="Description of the section")
    use_case: str = Field(..., description="Use case for the section")
    patterns: List[str] = Field(..., description="List of patterns for the section")
    technologies: List[str] = Field(..., description="List of technologies for the section")

class Sections(BaseModel):
    sections : list[Section]
    description : str = Field(..., description="Description of the section")

llm = ChatOpenAI(model="gpt-4")
planner = llm.with_structured_output(Sections)


class State(TypedDict):
    topic:str
    sections: list[Section]
    completed_sections: Annotated[
        list, operator.add
    ]  # All workers write to this key in parallel
    final_output:str    

class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[
        list, operator.add
    ]  # All workers write to this key in parallel

def llm_call(state: WorkerState):
    """Worker writes a section of the report"""

    # Generate section
    section = llm.invoke(
        [
            SystemMessage(
                content="Write a report section following the provided name and description. Include no preamble for each section. Use markdown formatting."
            ),
            HumanMessage(
                content=f"Here is the section name: {state['section'].name} and description: {state['section'].description}"
            ),
        ]
    )

    # Write the updated section to completed sections
    return {"completed_sections": [section.content]}

def asset_planner(state: State):
    """Plan sections for the given topic."""
    print(f"Debug: Input to planner.invoke: {state['topic']}")
    try:
        # Call the planner to generate sections
        decision = planner.invoke(
            [
                SystemMessage(
                    content="""You are an expert in SAP Integration. Identify all possible integration scenarios for the given topic."""
                ),
                HumanMessage(content=f"Generate scenarios for the topic: {state['topic']}"),
            ]
        )

        # Debug the output
        print(f"Debug: Output from planner.invoke: {decision.sections}")

        return {"sections": decision.sections}

    except Exception as e:
        # Log the error and re-raise it
        print(f"Error in asset_planner: {e}")
        raise ValueError(f"Failed to plan sections for the topic '{state['topic']}': {e}")

def assign_workers(state:State):
    """Assign a worker to each section in the plan"""
    # Kick off section writing in parallel via Send() API
    return [Send("llm_call", {"section": s}) for s in state["sections"]]

def synthesizer(state: State):
    """Synthesize full report from sections"""

    # List of completed sections
    completed_sections = state["completed_sections"]

    # Format completed section to str to use as context for final sections
    completed_report_sections = "\n\n---\n\n".join(completed_sections)

    return {"final_output": completed_report_sections}

# Build workflow
orchestrator_worker_builder = StateGraph(State)

# Add the nodes
orchestrator_worker_builder.add_node("orchestrator", asset_planner)
orchestrator_worker_builder.add_node("llm_call", llm_call)
orchestrator_worker_builder.add_node("synthesizer", synthesizer)

# Add edges to connect nodes
orchestrator_worker_builder.add_edge(START, "orchestrator")
orchestrator_worker_builder.add_conditional_edges(
    "orchestrator", assign_workers, ["llm_call"]
)
orchestrator_worker_builder.add_edge("llm_call", "synthesizer")
orchestrator_worker_builder.add_edge("synthesizer", END)

# Compile the workflow
orchestrator_worker = orchestrator_worker_builder.compile()



# ========== TESTING ==========
test_cases = [
    "Integration with SAP S/4HANA systems "
    
]

for i, test_input in enumerate(test_cases, 1):
    print(f"\n\nTEST CASE {i}: {test_input}")
    try:
        result = orchestrator_worker.invoke({"topic": test_input})
        print("\nRESULT:", result["final_output"])
    except Exception as e:
        print("ERROR:", str(e))
