@echo off

call venv\Scripts\activate
call pip install -r requirements.txt
call pip install ipykernel
call python -m ipykernel install --user --name=credit_venv --display-name "Python (credit_venv)"
jupyter notebook

call deactivate