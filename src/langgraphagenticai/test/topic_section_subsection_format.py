import os
from dotenv import load_dotenv
load_dotenv()
from typing import Annotated, List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
import operator
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from typing import Any
from langchain_groq import ChatGroq
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class PatternType(BaseModel):
    type: str
    patterns: List[str]

class Section(BaseModel):
    number: int = Field(..., description="Number of the section")
    name: str = Field(..., description="Name of the section")
    use_case: str = Field(..., description="Use case for the section")
    patterns: Dict[str, List[str]] = Field(..., description="Dictionary of pattern types and their patterns")
    technologies: List[str] = Field(..., description="List of technologies for the section")

class Sections(BaseModel):
    sections: List[Section]
    description: str = Field(..., description="Description of the overall topic")

#llm = ChatOpenAI(model="gpt-4")
llm = ChatGroq(model="qwen-2.5-32b")
planner = llm.with_structured_output(Sections)

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the input directory
input_dir = os.path.join(current_dir, "input")

""" # Define PDF file paths
pdf_files = [
    os.path.join(input_dir, "SAP_DMC_Integration_Guide_enUS.pdf"),
    os.path.join(input_dir, "Cloud Integration with SAP Integration Suite.pdf"),
    
] """

""" def load_and_process_pdfs(file_paths):
    documents = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        else:
            print(f"Warning: File not found - {file_path}")
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    return texts """

""" # Load and process PDFs
texts = load_and_process_pdfs(pdf_files) """

""" def initialize_faiss_db(texts):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore
 """
# Initialize FAISS vector store
#vectorstore = initialize_faiss_db(texts)

class State(TypedDict):
    topic: str
    sections: List[Section]
    completed_sections: Annotated[list, operator.add]
    final_output: str
    # vectorstore: Any    

class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]




def llm_call(state: WorkerState):
    """Worker writes a section of the report with specific subsections"""
    section = state['section']
    
    patterns_formatted = ""
    for pattern_type, pattern_list in section.patterns.items():
        patterns_formatted += f"  o {pattern_type}:\n"
        patterns_formatted += "".join([f"    - {pattern}\n" for pattern in pattern_list])

    full_section = f"""{section.number}. {section.name}

    - Use Case: {section.use_case}

    - Patterns:
        {patterns_formatted}
    - Technologies:
        {chr(10).join([f"  o {tech}" for tech in section.technologies])}
    """

    return {"completed_sections": [full_section]}

def asset_planner(state: State):
    print(f"Debug: Input to planner.invoke: {state['topic']}")
    try:
        #relevant_docs = state['vectorstore'].similarity_search(state['topic'], k=5)
        #context = "\n".join([doc.page_content for doc in relevant_docs])

        decision = planner.invoke([
            SystemMessage(content=f"""You are an expert in SAP Integration. Identify all possible integration scenarios for the given topic. 
            
            
            For each scenario, provide:
            1. A number
            2. A name
            3. A use case (1-2 sentences explaining the scenario)
            4. Patterns (as a dictionary with 'Real-time' and 'Batch' as keys. Provide a list of specific patterns for each key. If no patterns are applicable, provide an empty list)
            5. Technologies used (as a list of specific technologies, tools, or protocols)
            
            Ensure that the output is valid JSON, suitable for parsing into a Python dictionary.
            """),
            HumanMessage(content=f"Generate scenarios for the topic: {state['topic']}")
        ])

        print(f"Debug: Output from planner.invoke: {decision.sections}")
        return {"sections": decision.sections}

    except Exception as e:
        print(f"Error in asset_planner: {e}")
        raise ValueError(f"Failed to plan sections for the topic '{state['topic']}': {e}")

    
def assign_workers(state: State):
    """Assign a worker to each section in the plan"""
    return [Send("llm_call", {"section": s}) for s in state["sections"]]

def synthesizer(state: State):
    """Synthesize full report from sections"""
    completed_sections = state["completed_sections"]
    completed_report_sections = "\n\n".join(completed_sections)
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
    "What are the different functional integration scenarios w.r.t SAP S4 Hana system?"
]

for i, test_input in enumerate(test_cases, 1):
    print(f"\n\nTEST CASE {i}: {test_input}")
    try:
        result = orchestrator_worker.invoke({
            "topic": test_input
            
        })
        print("\nRESULT:", result["final_output"])
    except Exception as e:
        print("ERROR:", str(e))
