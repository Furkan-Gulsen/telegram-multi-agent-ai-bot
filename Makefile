.PHONY: install run clean venv activate

VENV_NAME=venv
PYTHON=python3
PIP=pip3

venv:
	$(PYTHON) -m venv $(VENV_NAME)
	@echo "Virtual environment '$(VENV_NAME)' created. Run 'source venv/bin/activate' to activate it."

activate:
	@echo "To activate the virtual environment, run: source venv/bin/activate"

install: venv
	. ./$(VENV_NAME)/bin/activate && $(PIP) install -r requirements.txt

run:
	@if [ -d "$(VENV_NAME)" ]; then \
		. ./$(VENV_NAME)/bin/activate && python main.py; \
	else \
		echo "Virtual environment not found. Run 'make install' first."; \
		exit 1; \
	fi

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".DS_Store" -delete

clean-venv: clean
	rm -rf $(VENV_NAME)