"""
This module provides a utility function for constructing prompts used in a 
retrieval-augmented generation (RAG) system for M.Tech admissions.

The prompt is loaded from an external template file (prompt_template.txt) to allow 
easy modification and configuration. The template instructs an AI assistant to 
generate clear and concise responses based on contextual information and a 
user-provided question.
"""


import os
from jinja2 import Template

def build_prompt(context: str, question: str) -> str:
    """
    Builds a prompt using a Jinja2 template located in the same folder.
    """
    # Path to the template file relative to this script's location
    template_path = os.path.join(os.path.dirname(__file__), "prompt_template.txt")
    
    with open(template_path, "r") as file:
        template_str = file.read()

    template = Template(template_str)
    return template.render(context=context, question=question)