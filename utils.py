import re
import logging
import subprocess
from pathlib import Path

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
        "-v", f"{Path(script_path).parent}:/manim",
        "manimcommunity/manim", "manim",
        Path(script_path).name, scene_name, "-ql", "--format=mp4",
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
