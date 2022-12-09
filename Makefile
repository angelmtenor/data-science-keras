# Development/Edition

setup:
	@echo "Installing Environment"
	pip install -r requirements-dev.txt
	pip install -e .

qa:
	@echo "Running QA"
	pre-commit run --all-files

clean:
	@if [ -s .\\.git\\hooks ]; then rmdir .\\.git\\hooks /q /s; fi
# @if [ -s poetry.lock ]; then rm poetry.lock; fi
