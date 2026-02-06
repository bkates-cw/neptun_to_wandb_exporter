import os

from neptune import management

api_token = os.environ.get("NEPTUNE_API_TOKEN")
if not api_token:
    print("Error: NEPTUNE_API_TOKEN environment variable not set")
    exit(1)

projects = management.get_project_list(api_token=api_token)
for p in projects:
    print(p)
