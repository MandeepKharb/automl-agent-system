from langgraph.graph import StateGraph, END
from state.ml_state import MLState
from agents.data_analyst import data_analyst_agent
from agents.feature_engineer import feature_engineer_agent
from agents.model_trainer import model_trainer_agent
from agents.report_writer import report_writer_agent

def build_workflow():
    workflow = StateGraph(MLState)

    # Add all 4 agent nodes
    workflow.add_node("data_analyst", data_analyst_agent)
    workflow.add_node("feature_engineer", feature_engineer_agent)
    workflow.add_node("model_trainer", model_trainer_agent)
    workflow.add_node("report_writer", report_writer_agent)

    # Define sequential flow
    workflow.set_entry_point("data_analyst")
    workflow.add_edge("data_analyst", "feature_engineer")
    workflow.add_edge("feature_engineer", "model_trainer")
    workflow.add_edge("model_trainer", "report_writer")
    workflow.add_edge("report_writer", END)

    return workflow.compile()