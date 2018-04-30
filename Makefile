.PHONY: update

poetry := poetry

get_poetry:
	pip install poetry

update:
	$(poetry) update

install:
	$(poetry) install

test: dev
	$(poetry) run pytest --cov=triflow --cov-report=html -p no:warnings

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
	$(poetry) run pip install ipython jupyter-contrib-nbextensions
	$(poetry) run $(MAKE) -C source_doc notebooks
	$(poetry) run $(MAKE) -C source_doc epub
	$(poetry) run $(MAKE) -C source_doc latexpdf
	$(poetry) run sphinx-build -b html source_doc/source docs
