SOURCES = main.py

all: $(SOURCES)
	python3 $(SOURCES)

test:
	python3 test.py

clean:
	rm -rf __pycache__ data/test_prediction.csv