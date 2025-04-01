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
        You are an expert in generating **high-quality Manim scripts** for mathematical and data visualizations. 
        
        **Objective:**  
        Generate a Manim script strictly based on the structured plan and user input. Follow all provided corrections and enhancements while ensuring smooth animations and clear explanations.

        ---
        
        **Input Details:**  
        - **User Request:** {user_input}  
        - **Planned Steps:**  
          {steps}  
        - **Improvements Requested:**  
          {improvement_suggestions}  
        - **Specific Fixes to Apply:**  
          {error_fixes}  
        
        ---
        
        **Strict Requirements:**  
        ✅ Use **Blue** for data representation and **Red** for results.  
        ✅ Ensure smooth animations with **0.5s transitions** for a polished experience.  
        ✅ Implement everything within **a single Scene class**.  
        🚫 **Do NOT use static images** like "house.png" or "scatter_example.jpg".  
        
        ---
        
        **Critical Note:**  
        ⚠️ **STRICTLY AVOID previous mistakes** mentioned below. Apply all specified fixes and ensure the script does NOT repeat those errors.  
        
        **Prevention Guide:**  
        {prevention_guide}
        
        **Final Instruction:**  
        - Output **only** the Manim script enclosed in triple backticks (```).  
        - Do not add explanations, extra text, or comments outside of the script.  
    """),
    ("human", "Generate the Manim script for {user_input}")
])

observe_prompt = ChatPromptTemplate.from_messages([
    ("system", """
        You are an expert in **Manim script debugging and evaluation**. Your task is to analyze the execution results and provide precise feedback.

        ---
        
        **Input Details:**  
        - **Concept:** {user_input}  
        - **Execution Status:** {status}  
        - **Error (if any):** {last_error}  
        - **Script Content:**  
        ```  
        {script_content}  
        ```  
        
        ---
        
        **Your Tasks:**  
        
        1️⃣ **If errors exist:**  
           - **Identify the exact root cause** (e.g., `"name 'random' is not defined"`).  
           - Provide precise fixes in the following structured format:  
             - **ADD:** `<code to add>` (e.g., `"import random"`)  
             - **REPLACE:** `<old code>` → `<new code>`  
             - **REMOVE:** `<code to remove>`  
           - Each fix should be on a **separate line**.  
           - **Do NOT suggest improvements here—focus ONLY on error fixes.**  
           
        2️⃣ **If the script executes successfully but needs improvements:**  
           - Compare the script with the original **plan** and **requirements**.  
           - Identify **missing, unclear, or ineffective parts**.  
           - Suggest **precise educational improvements** to enhance clarity and impact.  
           
        3️⃣ **If the script is correct and well-structured:**  
           - Respond with **"APPROVED"** (without any additional comments).  
        
        ---
        
        **Strict Response Format:**  
        - If errors exist:  
          ```
          ERROR FIXES:
          ADD: <fix1>
          REPLACE: <fix2>
          REMOVE: <fix3>
          ```  
          
        - If improvements are needed:  
          ```
          IMPROVEMENTS:
          <suggestion1>
          <suggestion2>
          ```  
          
        - If correct: `"APPROVED"`  
    """),
    ("human", "Review the execution of the Manim script")
])
