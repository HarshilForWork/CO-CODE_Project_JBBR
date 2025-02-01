import subprocess

# Define scripts in execution order (app.py runs last)
scripts = [
    "Streamlit/nlp_singleton.py",
    "Streamlit/document_processor.py",
    "Streamlit/difficulty_analyzer.py",
    "Streamlit/topic_analyzer.py",
    "Streamlit/distractor_generator.py",
    "Streamlit/mcq.py",
    "Streamlit/mcq_generator.py",
    "Streamlit/app.py"  # Last script to run
]

for script in scripts:
    print(f"Running {script}...")
    subprocess.run(["python", script])
