.PHONY: update

poetry := $(HOME)/.local/bin/poetry

get_poetry:
	pip install poetry --user

install:
	$(poetry) install

test: dev
	$(poetry) run pytest --cov=triflow --cov-report=html

lint: dev
	$(poetry) run pylama

isort: dev
	$(poetry) run "sh -c 'isort --recursive . '"

dev: get_poetry install
	$(poetry) run pip install -e .

build: dev doc
	$(poetry) build

publish: dev doc
	$(poetry) publish

doc:
	$(MAKE) -C source_doc notebooks
	$(MAKE) -C source_doc epub
	$(MAKE) -C source_doc latexpdf
	sphinx-build -b html source_doc/source docs
