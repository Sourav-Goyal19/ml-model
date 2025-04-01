import sqlite3
from typing import Dict, List
import logging

class ErrorMemory:
    def __init__(self, db_path: str = "manim_errors.db"):
        self.conn = sqlite3.connect(db_path)
        self._init_db()
    
    def _init_db(self):
        try:
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS error_knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                error_summary TEXT NOT NULL,
                solution TEXT NOT NULL,
                example_code TEXT,
                occurrences INTEGER DEFAULT 1,
                UNIQUE(error_summary, solution)
            )
            """)
            self.conn.commit()
        except sqlite3.Error as e:
            logging.error(f"Database initialization failed: {str(e)}")
            raise
        
    def analyze_error(self, raw_error: str, faulty_code: str, llm) -> Dict[str, str]:
        error_lines = raw_error.strip().split('\n')
        error_lines.reverse()
        exact_error = next(
            (line.strip() for line in error_lines 
             if line.strip().startswith(('NameError', 'AttributeError', 'ValueError', 'TypeError'))),
            raw_error[-500:]
        )
        relevant_code = faulty_code[-500:] if len(faulty_code) > 500 else faulty_code
        solution_prompt = f"""
        Fix this EXACT Manim error by providing ONLY the corrected code change:
        Error: {exact_error}
        Relevant Code:
        {relevant_code}

        Respond with ONE of:
        - "ADD: <Main Error>"
        - "REPLACE: <old line> WITH <new line>"

        Example Fixes:
        1. Error: "TypeError: getter() takes 1 positional argument but 2 were given"
        Fix: "REPLACE: `obj.getter(x,y)` WITH `obj.getter(point=(x,y))`"

        2. Error: "NameError: name 'np' is not defined"
        Fix: "ADD: import numpy as np"
        
        3. Error: "AttributeError: 'Mobject' has no attribute 'animate'"
        Fix: "REPLACE: `mobject.animate` WITH `mobject.shift`"
        """
        solution = llm.invoke(solution_prompt).content.strip()
        return {"summary": exact_error, "solution": solution}

    def record_error(self, raw_error: str, faulty_code: str, llm) -> str:
        analysis = self.analyze_error(raw_error, faulty_code, llm)
        try:
            self.conn.execute("""
            INSERT OR REPLACE INTO error_knowledge 
            (error_summary, solution, example_code, occurrences)
            VALUES (?, ?, ?, COALESCE(
                (SELECT occurrences FROM error_knowledge 
                 WHERE error_summary = ? AND solution = ?), 0) + 1)
            """, (
                analysis["summary"],
                analysis["solution"],
                faulty_code[-200:],
                analysis["summary"],
                analysis["solution"]
            ))
            self.conn.commit()
            return analysis["summary"]
        except sqlite3.Error as e:
            logging.error(f"Failed to record error: {str(e)}")
            return analysis["summary"]

    def get_prevention_guide(self) -> List[Dict]:
        try:
            cursor = self.conn.execute("""
            SELECT error_summary, solution, occurrences 
            FROM error_knowledge 
            ORDER BY occurrences DESC
            """)
            return [
                {"summary": row[0], "solution": row[1], "count": row[2]}
                for row in cursor.fetchall()
            ]
        except sqlite3.Error as e:
            logging.error(f"Failed to get prevention guide: {str(e)}")
            return []
    
    def close(self):
        try:
            self.conn.close()
        except sqlite3.Error as e:
            logging.error(f"Failed to close database: {str(e)}")
