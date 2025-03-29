import os
import requests
from langchain import hub
from links2 import MANIM_MODULES
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.agents import tool
from typing import List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor

load_dotenv()

os.environ["JINA_API_KEY"] = os.getenv("JINA_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

prompt_template = hub.pull("hwchase17/react")

@tool
def module_documentation_extractor(modules: str = None):
    """Returns the documentation for the specified Modules."""
    if not modules:
        return []
    
    content = []
    for module in modules.split(","):
        module = module.strip()
        path = f"dataset/{module}.txt"
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content.append(f.read())
        except Exception as e:
            content.append(f"Error reading {module}: {str(e)}")
    return content

tools = [module_documentation_extractor]

agent = create_react_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

class State(BaseModel):
    user_input: str = ""
    text_script: str = ""
    scenes: List[Dict[str, str]] = []  
    scene_descriptions: List[str] = []  
    scene_codes: List[str] = []  
    finalized_code_list: List[str] = []  
    final_code: str = "" 
    current_index: int = 0  


text_script_prompt = ChatPromptTemplate.from_messages([
    ("system", 
    """
        You are a Machine Learning Professor with decades of experience in teaching and explaining complex concepts to beginners. Your task is to:
        1. Create a clear, engaging, and concise explanation of the requested algorithm.
        2. Structure the explanation as a narrative that flows logically from introduction to conclusion.
        3. Use simple language suitable for students new to machine learning.
        4. Highlight key steps or concepts that can be visualized effectively.
        5. Avoid overly technical jargon unless simplified.
        Return a single, cohesive paragraph that serves as the script for a video explanation.
    """),
    ("human",
    """Generate an explanation script for the following query: {user_input}""")
])

scene_division_prompt = ChatPromptTemplate.from_messages([
    ("system",
    """
        You are an expert in breaking down educational scripts into visually distinct scenes for animation using Manim. Your role is to:
        1. Analyze the provided script and identify logical segments that can be visualized as separate scenes.
        2. Ensure each scene represents a clear, standalone visual state (e.g., showing data, animating a process).
        3. Keep scenes concise and focused, suitable for animation.
        4. Return ONLY a list of scene titles, one per line, separated by "\\n".
        
        Example output for a script about linear regression:
        Introduction to Linear Regression\nVisualizing the Data\nFitting the Line\nConclusion
    """),
    ("human",
    """Divide the following script into scenes: {text_script}""")
])

scene_description_prompt = ChatPromptTemplate.from_messages([
    ("system",
    """
        You are a Machine Learning engineer and Manim expert tasked with designing visual descriptions for animation scenes. For each scene:
        1. Create a detailed description of the visuals that should accompany the script segment.
        2. Focus on geometric transformations, data visualizations, or process animations that enhance understanding.
        3. Specify key elements like graphs, points, lines, or text labels that should appear.
        4. Ensure the description is concrete and animation-friendly.
        5. Keep the full script context in mind for continuity.
    """),
    ("human",
    """
        Generate a visual description for the scene titled '{scene_title}' based on this script segment: '{scene_content}'. Full script context: {text_script}
    """)
])

process_step_prompt = ChatPromptTemplate.from_messages([
    ("system",
    """
        You are a Manim expert specializing in ML algorithm visualizations. When generating code for the scene '{scene_title}', ensure you:
        1. Use appropriate Manim objects (e.g., Axes, Dots, Lines) to reflect the scene description.
        2. Create smooth, beginner-friendly animations that illustrate the concept.
        3. Add clear labels and annotations using the Text function (never Tex).
        4. Use a consistent color scheme (e.g., blue for data, red for predictions).
        5. Implement proper scaling and positioning for clarity on screen.
        6. Handle edge cases (e.g., empty data) gracefully.
        7. Match the animation duration to approximately 30-60 seconds per scene unless specified.
        8. Include all necessary imports and define a complete Scene class.
        9. Don't use any static image by thinking that user will replace that, you will code run directly, so it must be error free.
    """),
    ("human",
    """
        Generate production-quality Manim code for the scene described as: '{scene_description}'. Requirements:
        - Use your own custom random data if needed.
        - Ensure smooth timing between animations.
        - Return ONLY the implementation code.
    """)
])

refine_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert in Manim and Python programming, tasked with refining Manim visualization code for compatibility with version 0.19.0.
            You are provided with a list of Manim documentation Modules. Your job is to:
            1. Analyze the given code to identify which Manim classes, methods, or animations are used.
            2. Select the relevant Modules only and only from the provided list that correspond to these elements.
            3. Use the tool to fetch the latest documentation from only and only the given below Modules.
                Here is the full list of Manim documentation modules:
                {url_list}
            4. Refine the code by:
                - Replacing deprecated methods with updated equivalents based on the fetched documentation.
                - Optimizing animations for smoothness and scalability.
                - Ensuring proper syntax and best practices.
                - Replace Tex with Text in manim
            5. Return ONLY the corrected code followed by a line listing the Modules you used (e.g., 'Referenced modules: [module1], [module2]').

            """,
        ),
        (
            "human",
            """Refine the following Manim code to ensure compatibility with version 0.19.0:
            
            ```python
            {code}
            ```

            Use the tool to validate against the latest documentation

            Replace Tex with Text in manim
            
            Apply necessary fixes, optimizations, and improvements while keeping the animations visually effective.
            """,
        ),
    ]
)

script_integration_prompt = ChatPromptTemplate.from_messages([
    ("system",
    """
        You are a Manim expert tasked with integrating multiple scene codes into a single, error-free Manim script. Your role is to:
        1. Combine the provided list of scene codes into one cohesive script.
        2. Import all necessary Manim modules exactly once at the top.
        3. Define a single Scene class that runs all scenes sequentially with smooth transitions (e.g., FadeIn, FadeOut).
        4. Ensure no syntax errors or runtime issues occur.
        5. Maintain the original animations and timing from each scene.
        6. Add brief transition animations (e.g., 0.5-second fades) between scenes for continuity.
    """),
    ("human",
    """
        Integrate the following list of Manim scene codes into a single script with proper imports and transitions: {finalized_code_list}
    """)
])


def text_script_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("Entering text_script_node")
    chain = text_script_prompt | llm
    state["text_script"] = chain.invoke({"user_input": state["user_input"]}).content.strip()
    print(f"Text script generated: {state['text_script']}")
    return state

def scene_division_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("Entering scene_division_node")
    chain = scene_division_prompt | llm
    content = chain.invoke({"text_script": state["text_script"]}).content.strip()
    if isinstance(content, list):
        content = " ".join(str(item) for item in content)
    scene_titles = [title.strip() for title in content.split("\n") if title.strip()]
    if not scene_titles:
        scene_titles = ["Default Scene"]
    script_lines = state["text_script"].split(". ")
    lines_per_scene = max(1, len(script_lines) // len(scene_titles))
    state["scenes"] = [
        {"title": title, "content": ". ".join(script_lines[i * lines_per_scene:(i + 1) * lines_per_scene]).strip()}
        for i, title in enumerate(scene_titles)
    ]
    print(f"Scenes divided: {state['scenes']}")
    return state

def scene_description_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print(f"Entering scene_description_node. Current index: {state['current_index']}")
    chain = scene_description_prompt | llm
    current_scene = state["scenes"][state["current_index"]]
    description = chain.invoke({
        "scene_title": current_scene["title"],
        "scene_content": current_scene["content"],
        "text_script": state["text_script"]
    }).content.strip()
    if len(state["scene_descriptions"]) <= state["current_index"]:
        state["scene_descriptions"].append(description)
    else:
        state["scene_descriptions"][state["current_index"]] = description
    print(f"Scene description generated: {state['scene_descriptions'][state['current_index']]}")
    return state

def process_step_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print(f"Entering process_step_node. Current index: {state['current_index']}")
    chain = process_step_prompt | llm
    description = state["scene_descriptions"][state["current_index"]]
    code = chain.invoke({"scene_title": state["scenes"][state["current_index"]]["title"], "scene_description": description}).content.strip()
    if len(state["scene_codes"]) <= state["current_index"]:
        state["scene_codes"].append(code)
    else:
        state["scene_codes"][state["current_index"]] = code
    print(f"Scene code generated: {state['scene_codes'][state['current_index']]}")
    return state

def refine_step_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print(f"Entering refine_step_node. Current index: {state['current_index']}")

    code = state["scene_codes"][state["current_index"]]

    modules = "\n".join([f"{module}" for module in MANIM_MODULES])

    formatted_prompt = refine_prompt.format_messages(code=code, url_list=modules)

    system_message = formatted_prompt[0].content
    human_message = formatted_prompt[1].content
    combined_input = f"{system_message}\n\nHuman request: {human_message}"
    
    agent_input = {"input": combined_input}
    response = agent_executor.invoke(agent_input)
    
    refined_code = response["output"]
    if len(state["finalized_code_list"]) <= state["current_index"]:
        state["finalized_code_list"].append(refined_code)
    else:
        state["finalized_code_list"][state["current_index"]] = refined_code
    print(f"Scene code refined. Code added to finalized_code_list.")
    state["current_index"] += 1
    return state

def script_integration_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("Entering script_integration_node")
    chain = script_integration_prompt | llm
    state["final_code"] = chain.invoke({"finalized_code_list": state["finalized_code_list"]}).content.strip()
    print(f"Final code integrated.")
    return state

def should_continue(state: Dict[str, Any]) -> str:
    if state["current_index"] < len(state["scenes"]):
        print("Should continue: scene_description")
        return "scene_description"
    print("Should continue: script_integration")
    return "script_integration"


workflow = StateGraph(dict)
workflow.add_node("text_script", text_script_node)
workflow.add_node("scene_division", scene_division_node)
workflow.add_node("scene_description", scene_description_node)
workflow.add_node("process_step", process_step_node)
workflow.add_node("refine", refine_step_node)
workflow.add_node("script_integration", script_integration_node)

workflow.set_entry_point("text_script")
workflow.add_edge("text_script", "scene_division")
workflow.add_edge("scene_division", "scene_description")
workflow.add_edge("scene_description", "process_step")
workflow.add_edge("process_step", "refine")
workflow.add_conditional_edges(
    "refine", 
    should_continue, 
    {"scene_description": "scene_description", "script_integration": "script_integration"})
workflow.add_edge("script_integration", END)


def run_workflow(user_input: str) -> Dict[str, Any]:
    graph = workflow.compile()
    initial_state = State().dict()
    initial_state["user_input"] = user_input
    final_state = graph.invoke(initial_state)
    return final_state

if __name__ == "__main__":
    try:
        result = run_workflow("Explain linear regression")
        print("Final Code:\n", result["final_code"])
    except Exception as e:
        print(f"An error occurred: {str(e)}")