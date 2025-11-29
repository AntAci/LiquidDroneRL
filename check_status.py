from huggingface_hub import HfApi, login

import os
token = os.environ.get("HF_TOKEN")
repo_id = "iteratehack/team_22"

login(token=token)
api = HfApi()

try:
    # Get Space runtime info
    runtime = api.get_space_runtime(repo_id=repo_id)
    print(f"Stage: {runtime.stage}")
    print(f"Hardware: {runtime.hardware}")
    if runtime.stage == "RUNNING":
        print("Status: Training is RUNNING!")
    elif runtime.stage == "BUILDING":
        print("Status: Space is still BUILDING (installing dependencies)...")
    else:
        print(f"Status: {runtime.stage}")
        
except Exception as e:
    print(f"Error getting status: {e}")
