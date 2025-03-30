import os
import re
import logging
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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

think_prompt = ChatPromptTemplate.from_messages([
    ("system", """
        You are a Machine Learning professor and Manim expert tasked with creating a beginner-friendly Manim video script.
        Analyze the educational concept: {user_input}
        Consider: Key concepts to visualize, common student misconceptions, and effective teaching sequence.
        Output ONLY a concise paragraph of your reasoning.
    """),
    ("human", "Concept to explain: {user_input}")
])

plan_prompt = ChatPromptTemplate.from_messages([
    ("system", """
        Create a detailed plan for a Manim video explaining: {user_input}
        Based on this analysis: {reasoning}
        Include:
        1. Key visualizations needed
        2. Animation sequences
        3. Explanation flow
        4. Educational highlights
        Output ONLY numbered steps.
    """),
    ("human", "Create plan for: {user_input}")
])

action_prompt = ChatPromptTemplate.from_messages([
    ("system", """
        Generate/improve Manim script for: {user_input}
        Follow these steps: {steps}
        
        Error Fixes Needed:
        {error_fixes}
        
        Improvement Suggestions:
        {improvement_suggestions}
        
        Requirements:
        - Blue for data, red for results
        - Smooth animations (FadeIn, Create, etc)
        - 0.5s transitions
        - Single Scene class
        - 1-2 minute duration
        - Handle all edge cases
        Output ONLY the script in ``` marks.
    """),
    ("human", "Create/improve script for: {user_input}")
])

observe_prompt = ChatPromptTemplate.from_messages([
    ("system", """
        Analyze this Manim script execution:
        Input: {user_input}
        Execution Status: {status}
        Error: {last_error}
        
        Your tasks:
        1. If errors exist:
           - Diagnose root cause
           - Provide specific fixes (code snippets if possible)
           - Focus on runtime/syntax issues
     
           - **Important Note** - Don't suggest anything even IMPROVEMENTS, except fixing the ERROR(S)
        
        2. Else If successful but improvable:
           - Focus on educational clarity first, check on your own that what the created code is lacking according to the created plan:
            -----------Script: {script_content}---------
           - Don't messup any algorithm with other like linear regression with gradient descent.
           - Is it fulfilling its main content properly
           - Suggest specific improvements
        
        3. If perfect:
           - Output "APPROVED"
        
    """),
    ("human", "Review this execution")
])

def extract_code_block(text: str) -> str:
    pattern = r'```(?:python)?\s*\n(.*?)\n\s*```'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()

def extract_scene_name(script: str) -> str:
    match = re.search(r'class\s+(\w+)\s*\(Scene\):', script)
    return match.group(1) if match else "DefaultScene"

def run_manim_script(script_path: str, scene_name: str) -> tuple[bool, str]:
    command = [
        "docker", "run", "--rm",
        "-v", f"{os.path.dirname(script_path)}:/manim",
        "manimcommunity/manim:v0.18.0", "manim",
        os.path.basename(script_path), scene_name, "-ql", "--format=mp4",
        "--media_dir", "/manim/output"
    ]
    try:
        logging.info(f"Executing Manim script: {' '.join(command)}")
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=300,
            encoding="utf-8", errors="replace"
        )
        if result.returncode == 0:
            logging.info("Manim execution succeeded")
            return True, "Success"
        else:
            logging.error(f"Manim execution failed with code {result.returncode}")
            logging.error(f"Stderr: {result.stderr}")
            return False, result.stderr or "Unknown error"
    except subprocess.TimeoutExpired as e:
        logging.error(f"Manim execution timed out: {str(e)}")
        return False, "Timeout expired"
    except Exception as e:
        logging.error(f"Unexpected error in Manim execution: {str(e)}")
        return False, str(e)

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
    logging.info(f"Using error fixes: {state.get('error_fixes', 'None')}")
    logging.info(f"Using improvements: {state.get('improvement_suggestions', 'None')}")
    
    chain = action_prompt | llm
    script = chain.invoke({
        "user_input": state["user_input"],
        "steps": state["steps"],
        "error_fixes": state.get("error_fixes", "No fixes needed"),
        "improvement_suggestions": state.get("improvement_suggestions", "No improvements suggested")
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
    logging.info("Starting observe_node")
    logging.info(f"Current status: {state['status']}")
    
    chain = observe_prompt | llm
    analysis = chain.invoke({
        "user_input": state["user_input"],
        "status": state["status"],
        "last_error": state["last_error"],
        "script_content": state["script_content"]
    }).content.strip()
    
    state["attempts"] += 1
    logging.info(f"Observer analysis: {analysis}...")
    
    error_fixes = "None"
    improvements = "None"
    
    if "ERROR FIXES:" in analysis and "IMPROVEMENTS:" in analysis:
        error_section = analysis.split("ERROR FIXES:")[1].split("IMPROVEMENTS:")[0].strip()
        imp_section = analysis.split("IMPROVEMENTS:")[1].strip()
        error_fixes = error_section if error_section != "None" else "No fixes needed"
        improvements = imp_section if imp_section != "None" else "No improvements suggested"
        
        logging.info(f"Extracted error fixes: {error_fixes}...")
        logging.info(f"Extracted improvements: {improvements}...")
    elif analysis == "APPROVED":
        logging.info("Observer approved the script")
        return {
            "final_code": state["script_content"],
            "status": "approved"
        }
    
    return {
        "error_fixes": error_fixes,
        "improvement_suggestions": improvements,
        "observer_feedback": analysis
    }

def should_continue(state: AgentState) -> str:
    logging.info(f"Determining continuation. Status: {state.get('status')}, Attempts: {state.get('attempts', 0)}")
    
    if state.get("status") == "approved":
        logging.info("Workflow approved - ending")
        return "end"
    if state.get("status") == "error":
        logging.info("Errors detected - routing to fix_errors")
        return "fix_errors"
    if state.get("attempts", 0) >= 3:
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

app = workflow.compile()

def run_workflow(user_input: str) -> Dict[str, Any]:
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
    
    for step in app.stream(initial_state):
        for node, value in step.items():
            logging.info(f"Completed node: {node}")
            if "attempts" in value:
                # value["attempts"] += 1
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
    except Exception as e:
        logging.critical(f"Workflow failed: {str(e)}")
        raise
    