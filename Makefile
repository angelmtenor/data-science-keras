setup:
	@echo "Installing Environment"
	poetry install --all-extras

update:
	@echo "Updating Environment"
	poetry update
	pre-commit autoupdate

qa:
	@echo "Running QA"
	- mypy --install-types --non-interactive
	- pre-commit run --all-files

qa_fast:
	SKIP=pylint pre-commit run --all-files


test:
	pytest --cov=src

clean:
	@if [ -s .\\.git\\hooks ]; then rmdir .\\.git\\hooks /q /s; fi
	@if [ -s poetry.lock ]; then rm poetry.lock; fi

check:
ifeq ($(fast),True)
	@echo "yes"
else
	@echo "no"
endif
