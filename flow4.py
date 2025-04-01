import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from error_memory import ErrorMemory
from langgraph.graph import StateGraph, END
from typing import Dict, Any, TypedDict, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from utils import extract_code_block, extract_scene_name, run_manim_script
from prompts import think_prompt, plan_prompt, action_prompt, observe_prompt
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver();

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

error_memory = ErrorMemory()

def cleanup():
    error_memory.close()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

class AgentState(TypedDict):
    user_input: str
    reasoning: str
    steps: str
    script_content: str
    execution_result: str
    observer_feedback: str
    improvement_suggestions: str
    error_fixes: str
    final_code: str
    last_error: str
    attempts: int
    status: Literal["initial", "error", "success", "approved"]

def think_node(state: AgentState) -> Dict[str, str]:
    logging.info("Starting think_node")
    chain = think_prompt | llm
    reasoning = chain.invoke({"user_input": state["user_input"]}).content.strip()
    logging.info(f"Generated reasoning: {reasoning}...")
    return {"reasoning": reasoning}

def plan_node(state: AgentState) -> Dict[str, str]:
    logging.info("Starting plan_node")
    chain = plan_prompt | llm
    steps = chain.invoke({
        "user_input": state["user_input"],
        "reasoning": state["reasoning"]
    }).content.strip()
    logging.info(f"Generated steps: {steps}...")
    return {"steps": steps}

def action_node(state: AgentState) -> Dict[str, str]:
    logging.info("Starting action_node")
    
    prevention_guide = "\n".join(
        f"• {e['summary']} (Fix: {e['solution']})"
        for e in error_memory.get_prevention_guide()
    )

    print("\n\n**Prevention Guide: ", prevention_guide)
    
    chain = action_prompt | llm
    script = chain.invoke({
        "user_input": state["user_input"],
        "steps": state["steps"],
        "error_fixes": state.get("error_fixes", "No fixes needed"),
        "improvement_suggestions": state.get("improvement_suggestions", "No improvements suggested"),
        "prevention_guide": prevention_guide
    }).content.strip()
    
    script_content = extract_code_block(script)
    logging.info(f"Generated script (length: {len(script_content)} chars)")
    return {"script_content": script_content}

def execute_node(state: AgentState) -> Dict[str, str]:
    logging.info("Starting execute_node")
    script_path = str(Path.cwd() / "mymanim.py")

    logging.info(f"Writing script to {script_path}")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(state["script_content"])

    scene_name = extract_scene_name(state["script_content"])
    logging.info(f"Extracted scene name: {scene_name}")

    success, result = run_manim_script(script_path, scene_name)

    if success:
        logging.info("Execution completed successfully")
        return {
            "execution_result": result,
            "last_error": "",
            "status": "success"
        }
    else:
        logging.error(f"Execution failed: {result}")
        return {
            "execution_result": result,
            "last_error": result,
            "status": "error"
        }

def observe_node(state: AgentState) -> Dict[str, str]:
    current_attempt = state.get("attempts", 0) + 1
    logging.info(f"Attempt {current_attempt}. Status: {state['status']}")

    if state["status"] == "error":
        try:
            error_context = state["last_error"].strip().split('\n')[-15:]
            core_error = next(
                (line for line in reversed(error_context) 
                 if any(e in line for e in ["Error:", "Exception:", "Failed"])),
                state["last_error"]
            )
            
            error_summary = error_memory.record_error(
                raw_error='\n'.join(error_context),
                faulty_code=state["script_content"],
                llm=llm
            )
            logging.info(f"Recorded error: {error_summary}")
            
            prevention_guide = error_memory.get_prevention_guide()
            state["error_fixes"] = next(
                (e["solution"] for e in prevention_guide 
                 if core_error in e["summary"]),
                "No specific fix found"
            )
        except Exception as e:
            logging.error(f"Error recording failed: {str(e)}")
            state["error_fixes"] = "REPLACE: (see Manim documentation)"

    chain = observe_prompt | llm
    analysis = chain.invoke({
        "user_input": state["user_input"],
        "status": state["status"],
        "last_error": core_error if state["status"] == "error" else "",
        "script_content": state["script_content"]
    }).content.strip()

    error_fixes = "No fixes needed"
    improvements = "No improvements suggested"
    
    if "ERROR FIXES:" in analysis:
        error_fixes = analysis.split("ERROR FIXES:")[1].split("IMPROVEMENTS:")[0].strip()
        if not error_fixes.startswith(("ADD:", "REPLACE:")):
            error_fixes = state.get("error_fixes", "No valid fix generated")
    
    if "IMPROVEMENTS:" in analysis:
        improvements = analysis.split("IMPROVEMENTS:")[1].strip()
    
    if analysis == "APPROVED":
        return {
            "final_code": state["script_content"],
            "status": "approved",
            "attempts": current_attempt
        }

    return {
        "error_fixes": error_fixes,
        "improvement_suggestions": improvements,
        "observer_feedback": analysis,
        "attempts": current_attempt,
        "last_error": core_error if state["status"] == "error" else ""
    }

def should_continue(state: AgentState) -> str:
    logging.info(f"Determining continuation. Status: {state.get('status')}, Attempts: {state['attempts']}")

    if state.get("status") == "approved":
        logging.info("Workflow approved - ending")
        return "end"
    if state.get("status") == "error":
        logging.info("Errors detected - routing to fix_errors")
        return "fix_errors"
    if state["attempts"] >= 3:
        logging.warning("Max attempts reached - ending")
        return "end"

    logging.info("Routing to improve_quality")
    return "improve_quality"

workflow = StateGraph(AgentState)

workflow.add_node("think", think_node)
workflow.add_node("plan", plan_node)
workflow.add_node("action", action_node)
workflow.add_node("execute", execute_node)
workflow.add_node("observe", observe_node)

workflow.set_entry_point("think")
workflow.add_edge("think", "plan")
workflow.add_edge("plan", "action")
workflow.add_edge("action", "execute")
workflow.add_edge("execute", "observe")

workflow.add_conditional_edges(
    "observe",
    should_continue,
    {
        "end": END,
        "fix_errors": "action",
        "improve_quality": "action"
    }
)

config={
    "configurable": {
        "thread_id": 1
    }
}

app = workflow.compile(checkpointer=memory)

def run_workflow(user_input: str, thread_id: int = 1) -> Dict[str, Any]:
    logging.info(f"Starting workflow for input: {user_input}")

    initial_state = AgentState(
        user_input=user_input,
        reasoning="",
        steps="",
        script_content="",
        execution_result="",
        observer_feedback="",
        improvement_suggestions="",
        error_fixes="",
        final_code="",
        last_error="",
        attempts=0,
        status="initial"
    )

    for step in app.stream(initial_state, config=config):
        for node, value in step.items():
            logging.info(f"Completed node: {node}")
            if "attempts" in value:
                logging.info(f"Incremented attempt count to {value['attempts']}")

    final_state = value
    logging.info(f"Workflow completed with status: {final_state.get('status', 'unknown')}")

    return {
        "final_code": final_state.get("final_code", ""),
        "status": final_state.get("status", "unknown"),
        "attempts": final_state.get("attempts", 0),
        "feedback": final_state.get("observer_feedback", "")
    }

if __name__ == "__main__":
    try:
        logging.info("Starting main execution")
        result = run_workflow("Explain linear regression")

        print("\n=== Workflow Results ===")
        print(f"Attempts: {result['attempts']}")
        print(f"Final Status: {result['status']}")

        if result["final_code"]:
            print("\nFinal Script:")
            print(result["final_code"])

        if result["feedback"]:
            print("\nFinal Feedback:")
            print(result["feedback"])

        logging.info("Main execution completed successfully")
        print(app.get_state(config=config))
    except Exception as e:
        logging.critical(f"Workflow failed: {str(e)}")
        raise