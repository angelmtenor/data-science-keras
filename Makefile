setup:
	@echo "Installing Environment"
	poetry install --all-extras

qa:
	@echo "Running QA"
	- mypy --install-types --non-interactive
	- pre-commit run --all-files

test:
	pytest --cov=src

clean:
	@if [ -s .\\.git\\hooks ]; then rmdir .\\.git\\hooks /q /s; fi
	@if [ -s poetry.lock ]; then rm poetry.lock; fi
