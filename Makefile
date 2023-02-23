setup::
	@echo "Installing Environment"
	poetry install --all-extras

update::
	@echo "Updating Environment"
	poetry update
	pre-commit autoupdate

qa::
	@echo "Running complete QA"
	- mypy --install-types --non-interactive
	- pre-commit run --all-files

qa_fast::
	@echo "Running fast QA (skip: mypy, pylint & interrogate)"
	SKIP=mypy,pylint,interrogate pre-commit run --all-files


test::
	pytest --cov=src

clean::
	@if [ -s .\\.git\\hooks ]; then rmdir .\\.git\\hooks /q /s; fi
	@if [ -s poetry.lock ]; then rm poetry.lock; fi
