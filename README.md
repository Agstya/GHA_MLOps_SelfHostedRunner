# GHA_MLOps_SelfHostedRunner

# House Price Prediction Analysis Workflow

This GitHub Actions workflow automates the process of generating a house price prediction report and updating the report file in the repository. It utilizes various actions to handle different aspects of the task.

## Trigger Events

The workflow is triggered by the following events:

- Pull requests
- Pushes to any branch
- Manual workflow dispatch

## Jobs

The workflow consists of a single job named `build`.

### build Job

The `build` job runs on the `ubuntu-agastya` runner and performs the following steps:

1. **Checkout Code:** Checks out the code from the repository using the `actions/checkout@v2` action.

2. **Set Up Python:** Sets up the Python environment using the `actions/setup-python@v2` action with Python version `3.11`.

3. **Create Virtual Environment:** Creates a virtual environment named `myenv` using the `python -m venv myenv` command.

4. **Activate Virtual Environment:** Activates the virtual environment using the `source myenv/bin/activate` command.

5. **Install Dependencies:** Installs the required Python dependencies using the following commands:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

6. **Change Permissions of main.py:** Modifies the permissions of the `model/main.py` file to make it executable using the `chmod +x model/main.py` command.

7. **Run Python Script:** Executes the Python script `model/main.py`. This script is responsible for generating the house price prediction report.

8. **Display File Path and Report Contents:** Displays the file path of the generated report (`report.md`) and its contents using the `echo` command.

9. **Set Git Config:** Sets the Git configuration for the user to `actions@github.com` and the name to `GitHub Actions`.

10. **Create or Checkout Branch:** Creates a new branch named `experiment-1` or checks out an existing branch with the same name using the `git checkout experiment-1` command.

11. **Git Add, Commit, and Push:** Adds the updated report file (`report.md`) to the staging area, commits the changes with the message `Update report.md [skip ci]`, and pushes the changes to the `origin` remote repository on the `experiment-1` branch using the following commands:

```bash
git add report.md
git commit -m "Update report.md [skip ci]"
git push origin experiment-1
```

12. **Commit report.md:** Commits the changes made to the `report.md` file with the message `Update report.md [skip ci]` using the following commands:

```bash
git add report.md
git commit -m "Update report.md [skip ci]"
git push
```
This workflow effectively automates the process of generating and updating the house price prediction report, ensuring consistency and reproducibility.




