from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from state.ml_state import MLState
import json
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

def report_writer_agent(state: MLState) -> MLState:
    print("\n📝 Agent 4: Report Writer is working...")

    model_data = json.loads(state["model_result"])

    preprocessing_data = json.loads(state["preprocessing_result"])
    preprocessing_summary = {
        "processed_shape": preprocessing_data["processed_shape"],
        "final_columns": preprocessing_data["final_columns"],
        "target_column": preprocessing_data["target_column"]
    }

    # Use updated key names from generic model_tools
    model_summary = {
        "problem_type": model_data["problem_type"],
        "best_model": model_data["best_model"],
        "best_metric_label": model_data["best_metric_label"],
        "best_metric_value": model_data["best_metric_value"],
        "all_results": {
            name: {k: v for k, v in res.items() if k != "error"}
            for name, res in model_data["all_results"].items()
            if "error" not in res
        }
    }

    prompt = f"""
You are a technical report writer. Write a structured ML project report.

Problem Statement: {state["problem_statement"]}
Problem Type: {model_data["problem_type"]}
Data Analysis Findings: {state["analysis_result"]}
Preprocessing Summary: {json.dumps(preprocessing_summary, indent=2)}
Model Results: {json.dumps(model_summary, indent=2)}

Write a professional report with these sections:
1. Executive Summary
2. Dataset Overview
3. Data Preprocessing Steps
4. Model Comparison & Results
5. Best Model: {model_data["best_model"]} with {model_data["best_metric_label"]} of {model_data["best_metric_value"]}
6. Conclusion & Recommendations

Format it cleanly with clear section headers.
"""

    print("\n" + "=" * 60)
    print("📝  AGENT 4: REPORT WRITER")
    print("=" * 60)
    print(f"  📋 Problem type          : {model_data['problem_type'].upper()}")
    print(f"  🏆 Best model            : {model_data['best_model']}")
    print(f"  📈 {model_data['best_metric_label']:<20}   : {model_data['best_metric_value']}")
    print(f"\n  ✍️  Generating structured report...")

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        **state,
        "report": response.content,
        "current_agent": "report_writer"
    }