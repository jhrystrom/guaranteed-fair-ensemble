import subprocess

# Define your plots and args here
plots = {
    "plot_vision_frontiers": ["--methods=all"],
    "plot_improvements": [],
    "validate_competence": [],
}

for plot_file, args in plots.items():
    cmd = ["uv", "run", f"scripts/{plot_file}.py", *args]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
