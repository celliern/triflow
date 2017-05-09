.PHONY: clean-pyc clean-build

clean:
	pyenv uninstall -f triflow-test-3.6.1
	pyenv local --unset
	rm --force --recursive build/
	rm --force --recursive dist/
	rm --force --recursive .eggs/
	rm --force --recursive .cache/
	rm --force --recursive *.egg-info
	find . -name '*.pyc' -exec rm -f {} \;
	find . -type d -name "__pycache__" -delete
	find . -name '*.pyo' -exec rm -f {} \;
	find . -name '*~' ! -name '*.un~' -exec rm -f {} \;

pyenv:
	pyenv install -k 3.6.1
	pyenv virtualenv 3.6.1 triflow-test-3.6.1
	pyenv local triflow-test-3.6.1

env:
	pip install -Ur requirements.txt

init:
	pip install coveralls
	pip install isort
	pip install pytest-cov
	pip install pytest-pep8
	pip install pytest-xdist
	pip install pylama
	pip install recommonmark
	pip install .

lint:
	pylama

isort:
	sh -c "isort --recursive . "

build: clean
	python setup.py check
	python setup.py sdist
	python setup.py bdist_wheel

doc:
	$(MAKE) -C docs html
