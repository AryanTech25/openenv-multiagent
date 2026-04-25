import subprocess
import argparse

def compare(task, context):
    models = [
        ("HuggingFaceTB/SmolLM-135M", "local"),
        ("microsoft/phi-2", "local"),
        ("meta-llama/Meta-Llama-3-8B", "api")
    ]
    
    print("\n🚀 STARTING MULTI-MODEL COMPARISON...")
    
    for model_id, mode in models:
        cmd = [
            "./manager-worker-env/.venv/bin/python3", 
            "manager-worker-env/test_hf_workers.py",
            "--model", model_id,
            "--api-mode", mode,
            "--task", task,
            "--context", context
        ]
        subprocess.run(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Explain what a Multi-Agent System is.")
    parser.add_argument("--context", type=str, default="Be very brief.")
    args = parser.parse_args()
    
    compare(args.task, args.context)
