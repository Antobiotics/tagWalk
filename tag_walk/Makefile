
setup: clean
	python3 setup.py develop

dependencies-others:
	./lib/setup_requires.py

# Macosx, no cuda
dependencies-cpu: dependencies-others
	pip3 install tensorflow
	pip3 install http://download.pytorch.org/whl/torch-0.2.0.post3-cp36-cp36m-macosx_10_7_x86_64.whl 
	pip3 install torchvision

dependencies-gpu: dependencies-others
	pip3 install tensorflow-gpu
	pip3 install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl 
	pip3 install torchvision

install: clean
	python3 setup.py develop

clean:
	rm -rf env
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	rm -rf *.pyc
