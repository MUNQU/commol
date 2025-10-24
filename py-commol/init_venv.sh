pyenv local 3.13.2  # set python version
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install poetry
poetry install --extras dev, docs