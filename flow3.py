import os
import re
import logging
import subprocess
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Dict, Any, TypedDict, Annotated
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver

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
    final_code: str
    last_error: str
    attempts: int

think_prompt = ChatPromptTemplate.from_messages([
    ("system", """
        You are a Machine Learning professor and Manim expert tasked with creating a beginner-friendly Manim video script for '{user_input}'. Use a ReAct-like process:
        - Think: Reason about the key concepts, visuals (e.g., graphs, animations), and flow needed for clarity and engagement.
        - Focus on educational value and a logical progression of ideas.
        - Output ONLY a concise paragraph of your reasoning.
    """),
    ("human", "Reason about this query: {user_input}")
])

plan_prompt = ChatPromptTemplate.from_messages([
    ("system", """
        Based on your reasoning, outline a detailed plan to create a Manim video script for '{user_input}' that's engaging and educational for beginners.
        - Plan: Numbered, actionable steps covering explanation, visualization, and flow.
        - Include specific Manim elements (e.g., Axes, Dots, animations).
        - Output ONLY the steps, one per line.
    """),
    ("human", "Plan based on this reasoning: {reasoning}")
])

action_prompt = ChatPromptTemplate.from_messages([
    ("system", """
        You are a Manim expert creating a video script for '{user_input}' to teach beginners. Execute the steps provided with these guidelines:
        - Use Text (not Tex) for labels/titles, ensuring clarity.
        - Apply a consistent color scheme (blue for data, red for predictions/results).
        - Include smooth animations (e.g., FadeIn, Create) and 0.5-second fade transitions.
        - Define a single Scene class with all imports (e.g., `from manim import *`, `import random` if needed).
        - Target a 1-2 minute video, keeping it concise and executable.
        - Use random data if required, ensuring no undefined variables.
        - Output ONLY the script in ``` marks, with minimal comments.
    """),
    ("human", "Execute these steps for '{user_input}':\n{steps}")
])

observe_prompt = ChatPromptTemplate.from_messages([
    ("system", """
        You are a Manim expert reviewing the script for '{user_input}' based on its execution result. Your task is to:
        - If 'Success', check clarity and engagement; output 'APPROVED' if good, or a revised script in ``` marks if improvements are needed.
        - If an error or 'None', analyze the issue (e.g., syntax, missing imports), suggest fixes, and output a corrected script in ``` marks.
        - Ensure the script uses Text, follows the color scheme (blue data, red results), includes transitions, and is beginner-friendly.
    """),
    ("human", "Review this script:\n{script_content}\nExecution result: {execution_result}")
])


def extract_code_block(text: str) -> str:
    """Extracts code from triple-backtick formats."""
    pattern = r'```(?:python)?\s*\n(.*?)\n\s*```'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()

def extract_scene_name(script: str) -> str:
    """Extracts the class name of the Manim scene."""
    match = re.search(r'class\s+(\w+)\s*\(Scene\):', script)
    return match.group(1) if match else "DefaultScene"

def run_manim_script(script_path: str, scene_name: str) -> tuple[bool, str]:
    """Runs the Manim script in a Docker container with proper encoding."""
    command = [
        "docker", "run", "--rm",
        "-v", f"{os.path.dirname(script_path)}:/manim",
        "manimcommunity/manim:v0.18.0", "manim",
        os.path.basename(script_path), scene_name, "-ql", "--format=mp4",
        "--media_dir", "/manim/output"
    ]

    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=300,
            encoding="utf-8", errors="replace"
        )
        if result.returncode == 0:
            logging.info("Manim script executed successfully.")
            return True, "Success"
        else:
            logging.error("Manim execution failed: %s", result.stderr)
            return False, result.stderr or "Unknown error"
    except subprocess.TimeoutExpired as e:
        logging.error("Manim execution timed out: %s", e.stderr)
        return False, "Timeout expired"
    except Exception as e:
        logging.error("Unexpected error in Manim execution: %s", str(e))
        return False, str(e)

def think_node(state: AgentState) -> Dict[str, str]:
    think_chain = think_prompt | llm
    reasoning = think_chain.invoke({"user_input": state["user_input"]}).content.strip()
    logging.info("Reasoning: %s", reasoning)
    return {"reasoning": reasoning}

def plan_node(state: AgentState) -> Dict[str, str]:
    plan_chain = plan_prompt | llm
    steps = plan_chain.invoke({
        "reasoning": state["reasoning"],
        "user_input": state["user_input"]
    }).content.strip()
    logging.info("Planned Steps:\n%s", steps)
    return {"steps": steps}

def action_node(state: AgentState) -> Dict[str, str]:
    action_chain = action_prompt | llm
    script = action_chain.invoke({
        "user_input": state["user_input"],
        "steps": state["steps"]
    }).content.strip()
    script_content = extract_code_block(script)
    logging.info("Generated Script:\n%s", script_content)
    return {"script_content": script_content}

def execute_node(state: AgentState) -> Dict[str, str]:
    script_path = str(Path.cwd() / "mymanim.py")
    
    with open(script_path, "w") as f:
        f.write(state["script_content"])
    
    scene_name = extract_scene_name(state["script_content"])
    success, execution_result = run_manim_script(script_path, scene_name)
    
    if success:
        return {"execution_result": execution_result, "final_code": state["script_content"]}
    else:
        return {"execution_result": execution_result, "last_error": execution_result}

def observe_node(state: AgentState) -> Dict[str, str]:
    observe_chain = observe_prompt | llm
    observation = observe_chain.invoke({
        "user_input": state["user_input"],
        "script_content": state["script_content"],
        "execution_result": state["execution_result"]
    }).content.strip()
    
    logging.info("Observation: %s", observation[:200])
    
    if observation == "APPROVED":
        return {"final_code": state["script_content"]}
    elif observation.startswith("```"):
        return {"script_content": extract_code_block(observation)}
    else:
        return {"last_error": state["execution_result"]}

def should_retry(state: AgentState) -> str:
    if "final_code" in state and state["final_code"]:
        return "end"
    elif state.get("attempts", 0) >= 3:
        return "end"
    else:
        return "retry"

workflow = StateGraph(AgentState)

workflow.add_node("think", think_node)
workflow.add_node("plan", plan_node)
workflow.add_node("action", action_node)
workflow.add_node("execute", execute_node)
workflow.add_node("observe", observe_node)

workflow.add_edge("think", "plan")
workflow.add_edge("plan", "action")
workflow.add_edge("action", "execute")
workflow.add_edge("execute", "observe")

workflow.add_conditional_edges(
    "observe",
    should_retry,
    {
        "end": END,
        "retry": "action"
    }
)

workflow.set_entry_point("think")

app = workflow.compile()

def run_workflow(user_input: str) -> Dict[str, Any]:
    """Run the LangGraph workflow."""
    logging.info("Starting workflow for: %s", user_input)
    
    initial_state = AgentState(
        user_input=user_input,
        reasoning="",
        steps="",
        script_content="",
        execution_result="",
        final_code="",
        last_error="",
        attempts=0
    )
    
    for step in app.stream(initial_state):
        for node, value in step.items():
            logging.info(f"Node {node} completed")
            if "attempts" in value:
                value["attempts"] += 1
    
    final_state = value 
    return {
        "final_code": final_state.get("final_code", ""),
        "last_error": final_state.get("last_error", "")
    }

if __name__ == "__main__":
    try:
        result = run_workflow("Explain linear regression")
        print("Final Manim Script:\n", result["final_code"])
        if result["last_error"]:
            print("Last Error:\n", result["last_error"])
    except Exception as e:
        logging.critical("An error occurred: %s", str(e))
