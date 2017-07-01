
setup: clean
	python setup.py develop

setup-dev: clean
	virtualenv env
	./env/bin/pip install -U pip
	./env/bin/pip install -U setuptools
	./env/bin/python setup.py develop

install: clean
	python setup.py install

clean:
	rm -rf env
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
