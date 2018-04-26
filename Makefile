.PHONY: update

update:
	poetry update

test:
	pytest

lint:
	pylama

isort:
	sh -c "isort --recursive . "

build: doc
	poetry build

publish: doc
	poetry publish

doc:
	$(MAKE) -C source_doc notebooks
	$(MAKE) -C source_doc epub
	$(MAKE) -C source_doc latexpdf
	sphinx-build -b html source_doc/source docs
