import json, os, sys
f = os.path.expanduser('~/rldata/.vscode/settings.json')
s = json.load(open(f)) if os.path.exists(f) else {}
s['python.defaultInterpreterPath'] = sys.executable
json.dump(s, open(f, 'w'), indent=4)

# uv run --active python env_setup.py 
