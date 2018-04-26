.PHONY: update

get_poetry:
	pip install poetry --user

update:
	$(HOME)/.local/bin/poetry update

test:
	$(HOME)/.local/bin/poetry run pytest

lint:
	$(HOME)/.local/bin/poetry run pylama

isort:
	$(HOME)/.local/bin/poetry run "sh -c 'isort --recursive . '"

dev: get_poetry update
	$(HOME)/.local/bin/poetry run pip install -e .

build: doc
	$(HOME)/.local/bin/poetry build

publish: doc
	$(HOME)/.local/bin/poetry publish

doc:
	$(MAKE) -C source_doc notebooks
	$(MAKE) -C source_doc epub
	$(MAKE) -C source_doc latexpdf
	sphinx-build -b html source_doc/source docs
