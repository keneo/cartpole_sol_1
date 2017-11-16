
all: run

pima-indians-diabetes.data:
	wget http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data

run: pima-indians-diabetes.csv f.py
	python3 f.py



