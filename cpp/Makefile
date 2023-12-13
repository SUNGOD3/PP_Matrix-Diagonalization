# Write a Makefile of using g++ to compile Matrix.cpp, Matrix.hpp _matrix.cpp
CC = g++ -fopenmp  -mcmodel=medium -march=native
CFLAGS = -O3 -Wall -shared -std=c++11 -fPIC -ltbb
MKLINCLUDE = -I/usr/include/mkl 
PYINCLUDE = $(shell python3-config --includes)  $(shell python3 -m pybind11 --includes)
PYESUFFIX = $(shell python3-config --extension-suffix)
PYTESTPATH = $(shell pwd)/../test/
PYTHON = python3
TARGET=_matrix$(PYESUFFIX)

# PYTHONPATH := 
export PYTHONPATH := $(shell pwd):${PYTHONPATH}

all: $(TARGET)

$(TARGET): pybind.o _matrix.o _operation.o _eigen.o
	$(CC) $(CFLAGS) -o $(TARGET) $^

pybind.o: pybind.cpp
	$(CC) $(CFLAGS) -c pybind.cpp $(PYINCLUDE)

_matrix.o: _matrix.cpp _matrix.hpp
	$(CC) $(CFLAGS) -c _matrix.cpp $(PYINCLUDE)

_operation.o: _operation.cpp _operation.hpp _matrix.hpp
	$(CC) $(CFLAGS) -c _operation.cpp $(PYINCLUDE)

_eigen.o: _eigen.cpp _eigen.hpp _matrix.hpp
	$(CC) $(CFLAGS) -c _eigen.cpp $(PYINCLUDE)

.PHONY: test
test: $(TARGET)
	$(PYTHON) -m pytest -v ${PYTESTPATH}test.py

.PHONY: demo
demo: $(TARGET)
	# echo $(PYTHONPATH)
	$(PYTHON) ${PYTESTPATH}test.py

.PHONY: clean
clean:
	rm -rf *.o *.so
	rm -rf ${PYTESTPATH}__pycache__