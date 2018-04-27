.PHONY: update

poetry := $(HOME)/.local/bin/poetry

get_poetry:
	pip install poetry --user

update:
	$(poetry) update

test:
	$(poetry) run pytest --cov=triflow

lint:
	$(poetry) run pylama

isort:
	$(poetry) run "sh -c 'isort --recursive . '"

dev: get_poetry update
	$(poetry) run pip install -e .

build: doc
	$(poetry) build

publish: doc
	$(poetry) publish

doc:
	$(MAKE) -C source_doc notebooks
	$(MAKE) -C source_doc epub
	$(MAKE) -C source_doc latexpdf
	sphinx-build -b html source_doc/source docs
