import os
from langchain.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from typing import Dict, List, Any
from pydantic import BaseModel
from langchain.agents import tool
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from bs4 import BeautifulSoup
import requests
from links import MANIM_URLS

load_dotenv()

os.environ["JINA_API_KEY"] = os.getenv("JINA_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# llm = ChatAnthropic(
#     model_name="claude-3-5-sonnet-20240620",
#     api_key=os.getenv("CLAUDE_API_KEY"),
# )

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
)

@tool
def url_content_extractor(urls: str = None):
    """Fetch and extract content from specified URLs using BeautifulSoup."""
    if not urls:
        return []
    ur = urls.split(",");
    contents = []
    for url in ur:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            pre_tags = soup.find_all('pre')
            content = '\n'.join([pre.get_text(strip=True) for pre in pre_tags])
            contents.append(content)
        except Exception as e:
            contents.append(f"Error processing {url}: {str(e)}")
    
    return contents

prompt_template = hub.pull("hwchase17/react")

tools = [url_content_extractor]

agent = create_react_agent(llm, tools, prompt_template)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True);

class State(BaseModel):
    user_input: str = ""
    algorithm: str = ""
    steps: List[str] = []
    current_step_index: int = 0
    current_step_attempt: List[str] = []
    final_code: List[str] = []


identify_algorithm_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert ML algorithm identification system. Your role is to:
            1. Analyze user input to identify the most appropriate machine learning algorithm
            2. Consider factors like:
            - Problem type (classification, regression, clustering, etc.)
            - Data characteristics mentioned
            - Performance requirements
            - Implementation constraints
            3. Return ONLY the algorithm name without explanation
            4. Stick to well-established algorithms that can be effectively visualized""",
        ),
        (
            "human",
            """Based on this user request, identify the single most appropriate ML algorithm.
            If multiple algorithms could work, choose the most fundamental one that's easier to visualize.
            Request: {user_input}""",
        ),
    ]
)

plan_explanation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert in breaking down machine learning algorithms into clear, visualizable steps.
            For algorithm {algorithm}, create steps that:
            1. Start with data preparation/input
            2. Include all key mathematical transformations
            3. Show how the algorithm processes data iteratively
            4. Conclude with output/prediction generation
            5. Focus on aspects that can be effectively animated
            Each step should be concrete and visualizable.""",
        ),
        (
            "human",
            """Create a sequence of visualization-friendly steps for {algorithm}. Requirements:
                - Each step should represent a distinct visual state
                - Include data transformations that can be animated
                - Focus on geometric and mathematical operations
                - Ensure steps flow logically from input to output
                - Important Note-Return ONLY numbered steps, one per line""",
        ),
    ]
)

process_step_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a Manim expert specializing in ML algorithm visualization.
            When generating code for {algorithm}, ensure you:
            1. Use appropriate Manim objects (Axes, Dots, Lines, etc.)
            2. Create smooth animations between states
            3. Add clear labels and annotations
            4. Use consistent color schemes
            5. Implement proper scaling and positioning
            6. Handle edge cases gracefully
            7. Don't need to write any comment
            8. Always use Text function of manim instead of Tex
            """,
        ),
        (
            "human",
            """Generate production-quality Manim code for this step of {algorithm}:
            Step to visualize: {current_step}

                Requirements:
                - Include all necessary imports
                - Create a complete Scene class
                - Use meaningful variable names
                - Add comments explaining complex animations
                - Ensure proper timing between animations
                - Always use your own custom random data
                - Return ONLY the implementation code""",
        ),
    ]
)

refine_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert in Manim and Python programming, tasked with refining Manim visualization code for compatibility with version 0.19.0.
            You are provided with a list of Manim documentation URLs. Your job is to:
            1. Analyze the given code to identify which Manim classes, methods, or animations are used.
            2. Select the relevant URLs only from the provided list that correspond to these elements.
            3. Use the tool to fetch the latest documentation from only the below given URLs.
                Here is the full list of Manim documentation URLs:
                {url_list}
            4. Refine the code by:
                - Replacing deprecated methods with updated equivalents based on the fetched documentation.
                - Optimizing animations for smoothness and scalability.
                - Ensuring proper syntax and best practices.
                - Replace Tex with Text in manim
            5. Return ONLY the corrected code followed by a line listing the URLs you used (e.g., 'Referenced websites: [url1], [url2]').

            """,
        ),
        (
            "human",
            """Refine the following Manim code to ensure compatibility with version 0.19.0:
            
            ```python
            {code}
            ```

            Use the tool to validate against the latest documentation from 'https://docs.manim.community/en/stable/_modules/'.

            Replace Tex with Text in manim
            
            Apply necessary fixes, optimizations, and improvements while keeping the animations visually effective.
            """,
        ),
    ]
)


def identify_algorithm(state: Dict[str, Any]) -> Dict[str, Any]:
    print("Entering identify_algorithm node")
    chain = identify_algorithm_prompt | llm
    response = chain.invoke({"user_input": state["user_input"]})
    content = response.content
    if isinstance(content, list):
        content = " ".join(str(item) for item in content)
    state["algorithm"] = content.strip()
    print(f"Algorithm identified: {state['algorithm']}")
    return state

def plan_explanation(state: Dict[str, Any]) -> Dict[str, Any]:
    print("Entering plan_explanation node")
    chain = plan_explanation_prompt | llm
    response = chain.invoke({"algorithm": state["algorithm"]})
    content = response.content
    if isinstance(content, list):
        content = " ".join(str(item) for item in content)
    response_text = content
    steps = [step for step in response_text.split("\n") if step.strip()]
    if not steps:
        steps = ["No clear steps found. Please check the algorithm."]
    state["steps"] = steps
    print(f"Steps planned: {state['steps']}")
    return state

def process_step(state: Dict[str, Any]) -> Dict[str, Any]:
    print(f"Entering process_step node. Current step index: {state['current_step_index']}")
    chain = process_step_prompt | llm
    current_step = state["steps"][state["current_step_index"]]
    response = chain.invoke({"algorithm": state["algorithm"], "current_step": current_step})
    content = response.content
    if isinstance(content, list):
        content = " ".join(str(item) for item in content)
    if len(state["current_step_attempt"]) <= state["current_step_index"]:
        state["current_step_attempt"].append(content.strip())
    else:
        state["current_step_attempt"][state["current_step_index"]] = content.strip()
    print(f"Step processed: {state['current_step_attempt'][state['current_step_index']]}")
    return state

def refine_step(state: Dict[str, Any]) -> Dict[str, Any]:
    print(f"Entering refine node. Current step index: {state['current_step_index']}")
    
    code = state["current_step_attempt"][state["current_step_index"]]
    
    url_list = "\n".join([f"{url}" for url in MANIM_URLS])
    
    formatted_prompt = refine_prompt.format_messages(code=code, url_list=url_list)
    
    system_message = formatted_prompt[0].content
    human_message = formatted_prompt[1].content
    combined_input = f"{system_message}\n\nHuman request: {human_message}"
    
    agent_input = {"input": combined_input}
    response = agent_executor.invoke(agent_input)
    
    refined_code = response["output"]
    
    state["final_code"].append(refined_code.strip())
    print("Refined node code for step index", state["current_step_index"], ":")
    print("Content: ", refined_code)
    print(f"Step refined. Code added to final_code.")
    state["current_step_index"] += 1
    
    return state

def should_continue_steps(state: Dict[str, Any]) -> str:
    if state["current_step_index"] < len(state["steps"]) - 1:
        state["current_step_index"] += 1
        print("Should continue steps: process_step")
        return "process_step"
    print("Should continue steps: END")
    return END


workflow = StateGraph(dict) 
workflow.add_node("identify_algorithm", identify_algorithm)
workflow.add_node("plan_explanation", plan_explanation)
workflow.add_node("process_step", process_step)
workflow.add_node("refine", refine_step)

workflow.set_entry_point("identify_algorithm")
workflow.add_edge("identify_algorithm", "plan_explanation")
workflow.add_edge("plan_explanation", "process_step")
workflow.add_edge("process_step", "refine")
workflow.add_conditional_edges(
    "refine",
    should_continue_steps,
    { "process_step": "process_step", END: END},
)


def run_workflow(user_input: str) -> Dict[str, Any]:
    graph = workflow.compile()
    initial_state = {
        "user_input": user_input,
        "algorithm": "",
        "steps": [],
        "current_step_index": 0,
        "current_step_attempt":  [],
        "final_code": []
    }
    final_state = graph.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    try:
        result = run_workflow("Explain linear regression")
        print("Algorithm:", result["algorithm"])
        print("\nSteps:")
        for i, step in enumerate(result["steps"]):
            print(f"{i+1}. {step}")
        print("\nGenerated Code:")
        for i, code in enumerate(result["final_code"]):
            print(f"\nStep {i+1} visualization code:\n{code}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
