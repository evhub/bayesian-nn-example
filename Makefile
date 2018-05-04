.PHONY: run
run: setup
	python ./bnns_example.py

.PHONY: setup
setup:
	pip install numpy tensorflow edward

.PHONY: clean
clean:
	rm -rf ./log
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -delete

.PHONY: tensorboard
tensorboard:
	open http://localhost:6006
	tensorboard --logdir=./log
