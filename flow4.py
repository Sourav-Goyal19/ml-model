import os
import re
import logging
import subprocess
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Dict, Any
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

# Define Pydantic state model
class State(BaseModel):
    user_input: str = ""
    final_code: str = ""
    last_error: str = ""

# Enhanced Prompts
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
    ("human", "Review this script:\n{script}\nExecution result: {execution_result}")
])

def extract_code_block(text: str) -> str:
    """Extracts code from triple-backtick formats."""
    pattern = r'```(?:python)?\s*\n(.*?)\n\s*```'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()

def extract_scene_name(script: str) -> str:
    """Extracts the class name of the Manim scene."""
    match = re.search(r'class\s+(\w+)\s*$$ Scene $$:', script)
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
            encoding="utf-8", errors="replace"  # Handle encoding issues gracefully
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

def run_workflow(user_input: str, max_retries: int = 3) -> Dict[str, Any]:
    logging.info("Starting workflow for: %s", user_input)
    state = State(user_input=user_input).dict()

    # Think
    think_chain = think_prompt | llm
    reasoning = think_chain.invoke({"user_input": user_input}).content.strip()
    logging.info("Reasoning: %s", reasoning)

    # Plan
    plan_chain = plan_prompt | llm
    steps = plan_chain.invoke({"reasoning": reasoning, "user_input": user_input}).content.strip()
    logging.info("Planned Steps:\n%s", steps)

    # Action and Observe with retries
    action_chain = action_prompt | llm
    observe_chain = observe_prompt | llm
    script_path = str(Path.cwd() / "mymanim.py")

    script = None
    for attempt in range(max_retries):
        # Action
        script = action_chain.invoke({"user_input": user_input, "steps": steps}).content.strip()
        script_content = extract_code_block(script)
        logging.info("Generated Script (Attempt %d):\n%s", attempt + 1, script_content)  # Log full script

        # Save and run
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)

        scene_name = extract_scene_name(script_content)
        success, execution_result = run_manim_script(script_path, scene_name)

        if success:
            state["final_code"] = script_content
            logging.info("Workflow completed successfully.")
            return state

        # Observe and refine
        observation = observe_chain.invoke({
            "user_input": user_input,
            "script": script_content,
            "execution_result": execution_result
        }).content.strip()
        logging.warning("Observation (Attempt %d): %s", attempt + 1, observation[:200])

        if observation == "APPROVED":
            state["final_code"] = script_content
            logging.info("Workflow completed with AI approval.")
            return state
        elif observation.startswith("```"):
            script = observation  # Update for next iteration
        else:
            state["last_error"] = execution_result
            logging.warning("Observation didn't provide a corrected script.")

    state["last_error"] = execution_result if 'execution_result' in locals() else "Unknown error"
    logging.error("Max retries reached. Final script could not be validated.")
    return state

if __name__ == "__main__":
    try:
        result = run_workflow("Explain linear regression", max_retries=3)
        print("Final Manim Script:\n", result["final_code"])
        if result["last_error"]:
            print("Last Error:\n", result["last_error"])
    except Exception as e:
        logging.critical("An error occurred: %s", str(e))