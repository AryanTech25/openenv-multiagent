import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.task_library import TaskLibrary

def generate_data():
    library = TaskLibrary()
    data = []
    
    # Domain specific expert response templates
    templates = {
        "web_development": "Expert HTML/CSS/JS implementation with modern best practices.",
        "research": "Comprehensive analysis based on data-driven insights and market trends.",
        "software_engineering": "Robust, efficient code with proper error handling and documentation.",
        "product_management": "Strategic planning focusing on user needs, timelines, and business goals.",
        "academic_writing": "Formal, well-structured content with clear methodology and citations."
    }

    print(f"Generating multi-task data from {len(library.tasks)} task templates...")
    
    for task in library.tasks:
        domain_style = templates.get(task.task_type, "High-quality professional output.")
        
        for subtask in task.subtasks:
            # In a real scenario, you'd put actual expert outputs here.
            # For this exercise, we are creating the structure for the user to fill or for synthetic generation.
            example = {
                "text": f"Task: {subtask.description}\nContext: Domain: {task.task_type}. Instruction: {task.description}.\nOutput: {domain_style} This fulfills the requirement for {subtask.expected_output_format}."
            }
            data.append(example)
            
    output_path = "manager-worker-env/training/multitask_worker_data.jsonl"
    with open(output_path, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
    
    print(f"✓ Created {len(data)} training examples at {output_path}")

if __name__ == "__main__":
    generate_data()
