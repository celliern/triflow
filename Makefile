.PHONY: update

poetry := poetry

get_poetry:
	pip install poetry ${PIPARG}

clean:
	@rm -Rf *.egg .cache .coverage .tox build dist docs/build htmlcov .pytest_cache
	@find -depth -type d -name __pycache__ -exec rm -Rf {} \;
	@find -type f -name '*.pyc' -delete

update:
	$(poetry) update

install:
	$(poetry) install

test: clean dev
	$(poetry) run pytest --cov=triflow --cov-report=html --doctest-glob="*.rst"

lint: dev
	$(poetry) run pylama

notebook: dev
	$(poetry) run jupyter notebook

isort: dev
	$(poetry) run isort --recursive .

check: clean dev lint isort test

dev: get_poetry install
	$(poetry) run pip install -e .

build: dev doc
	$(poetry) build

publish: check build
	$(poetry) publish

doc:
	$(poetry) run pip install ipython jupyter-contrib-nbextensions
	$(poetry) run sphinx-apidoc -f -H "Module API" -o source_doc/source triflow/
	$(poetry) run $(MAKE) -C source_doc notebooks
	$(poetry) run $(MAKE) -C source_doc epub
	$(poetry) run $(MAKE) -C source_doc latexpdf
	$(poetry) run sphinx-build -b html source_doc/source docs
