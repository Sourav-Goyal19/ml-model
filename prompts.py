from langchain_core.prompts import ChatPromptTemplate

think_prompt = ChatPromptTemplate.from_messages([
    ("system", """
        You are a Machine Learning professor and an expert in Manim animations. Your task is to analyze and break down the educational concept provided.
        
        Concept: {user_input}
        
        Consider the following:
        - Key concepts that must be visually explained
        - Common misconceptions that students might have
        - The best sequence to introduce and explain these ideas visually
        
        Output ONLY a structured paragraph with key insights on how to present the topic effectively.
    """),
    ("human", "Explain the key visual aspects of {user_input}")
])

plan_prompt = ChatPromptTemplate.from_messages([
    ("system", """
        Using the reasoning provided below, create a structured plan for a Manim animation explaining {user_input}.
        
        Reasoning: {reasoning}
        
        Your plan should include:
        1. Essential visual components (graphs, equations, geometric representations, etc.)
        2. Step-by-step animation sequence
        3. Logical flow of explanation to maintain clarity
        4. Educational highlights that reinforce understanding
        
        Output ONLY in numbered list format.
    """),
    ("human", "Provide a detailed animation plan for {user_input}")
])

action_prompt = ChatPromptTemplate.from_messages([
    ("system", """
        Generate a Manim script based on the structured plan for: {user_input}
        
        Plan Steps:
        {steps}
        
        Previous Errors and Fixes to Avoid:
        {prevention_guide}
        
        Improvements Requested:
        {improvement_suggestions}
        
        Specific Fixes to Apply:
        {error_fixes}
        
        **Requirements for the script:**
        - Use Blue for data and Red for results
        - Ensure smooth animations with 0.5s transitions
        - Implement everything within a single Scene class
        - Avoid using static images like "house.png" or "scatter_example.jpg"
        
        **Important:** Incorporate the specific fixes provided to resolve previous errors.
        
        Output ONLY the Manim script enclosed within triple backticks (```).
    """),
    ("human", "Generate the Manim script for {user_input}")
])

observe_prompt = ChatPromptTemplate.from_messages([
    ("system", """
        Analyze the execution of the Manim script.
        
        Concept: {user_input}
        Execution Status: {status}
        Error (if any): {last_error}
        Script Content:
        ```{script_content}```
        
        Your tasks:
        1. If errors exist:
           - Identify the root cause (e.g., "name 'random' is not defined")
           - Provide fixes in the following format:
             - ADD: <code to add, e.g., "import random">
             - REPLACE: <old code> with <new code, e.g., "self.camera.frame" with "self.camera">
             - REMOVE: <code to remove>
           - Each fix on a new line
           - **Do NOT suggest improvements here—only fix errors.**
        
        2. If execution is successful but improvement is needed:
           - Compare the script with the original plan
           - Identify missing or unclear parts
           - Suggest improvements that enhance educational clarity
        
        3. If the script is correct and well-structured:
           - Respond with **"APPROVED"**.
        
        Format responses strictly as follows:
        - If errors exist: "ERROR FIXES:\n<fix1>\n<fix2>\n..."
        - If improvements are needed: "IMPROVEMENTS:\n<suggestion1>\n<suggestion2>\n..."
        - If correct: "APPROVED"
    """),
    ("human", "Review the execution of the Manim script")
])