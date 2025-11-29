from huggingface_hub import HfApi, login
import sys

import os
token = os.environ.get("HF_TOKEN")
repo_id = "iteratehack/team_22"

try:
    login(token=token)
    api = HfApi()
    
    # Check if repo exists, if not create it
    try:
        api.repo_info(repo_id=repo_id, repo_type="space")
        print(f"Space {repo_id} already exists.")
    except Exception:
        print(f"Creating Space {repo_id}...")
        api.create_repo(
            repo_id=repo_id, 
            repo_type="space", 
            space_sdk="docker",
            private=False # Or true? Hackathons usually public or private? Let's default to public or check user intent. 
            # Actually, usually private is safer for billing, but let's try public first or just default.
            # The prompt didn't specify, but usually spaces are public.
        )
        print(f"Space {repo_id} created.")

    # We also need to set the secret HF_TOKEN
    print(f"Setting secret HF_TOKEN in {repo_id}...")
    api.add_space_secret(repo_id=repo_id, key="HF_TOKEN", value=token)
    print("Secret set.")

except Exception as e:
    print(f"Error: {e}")
