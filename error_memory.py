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
        """Use AI to generate error summary and solution"""
        try:
            summary = llm.invoke(f"""
            Extract JUST THE CORE MANIM ERROR in one sentence:
            Error: {raw_error[:500]}
            Code: {faulty_code[:500]}
            
            Respond ONLY with the technical problem summary.
            Example: "Cannot animate NoneType objects"
            """).content.strip()

            solution = llm.invoke(f"""
            For this Manim error, provide JUST THE CODE SOLUTION:
            Error: {raw_error[:500]}
            
            Respond ONLY with the specific fix.
            Example: "Initialize mobject before animating"
            """).content.strip()

            return {"summary": summary, "solution": solution}
        except Exception as e:
            logging.error(f"Error analysis failed: {str(e)}")
            return {"summary": "Unknown error", "solution": "Check Manim documentation"}

    def record_error(self, raw_error: str, faulty_code: str, llm) -> str:
        """Store error and return summary"""
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
                faulty_code[:200],
                analysis["summary"],
                analysis["solution"]
            ))
            self.conn.commit()
            return analysis["summary"]
        except sqlite3.Error as e:
            logging.error(f"Failed to record error: {str(e)}")
            return analysis["summary"]

    def get_prevention_guide(self) -> List[Dict]:
        """Get top error patterns for prevention"""
        try:
            cursor = self.conn.execute("""
            SELECT error_summary, solution, occurrences 
            FROM error_knowledge 
            ORDER BY occurrences DESC
            LIMIT 5
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