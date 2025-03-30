from langchain_core.prompts import ChatPromptTemplate

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
        
        KNOWN ERROR PATTERNS TO AVOID:
        {prevention_guide}
        
        Steps: {steps}
        Fixes Needed: {error_fixes}
        Improvements: {improvement_suggestions}
        
        Requirements:
        - Blue=data, Red=results
        - Smooth animations
        - 0.5s transitions
        - Single Scene class
        
        Output ONLY the script in ``` marks.
    """),
    ("human", "Create script for: {user_input}")
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
