.PHONY: clean-pyc clean-build
TEST_PATHS=triflow docs tests README.rst

env:
	pip install -Ur requirements.txt

init: env
	pip install coveralls
	pip install pytest-cov
	pip install pytest-xdist
	pip install pylama
	pip install recommonmark
	pip install .

clean:
	rm --force --recursive build/
	rm --force --recursive dist/
	rm --force --recursive .eggs/
	rm --force --recursive .cache/
	rm --force --recursive *.egg-info
	find . -name '*.pyc' -exec rm -f {} \;
	find . -type d -name "__pycache__" -delete
	find . -name '*.pyo' -exec rm -f {} \;
	find . -name '*~' ! -name '*.un~' -exec rm -f {} \;

lint:
	pylama

isort:
	sh -c "isort --recursive . "

test:
	pytest

doc:
	$(MAKE) -C docs html

info:
	@python --version
	@pip --version

build: clean
	python setup.py check
	python setup.py sdist
	python setup.py bdist_wheel
