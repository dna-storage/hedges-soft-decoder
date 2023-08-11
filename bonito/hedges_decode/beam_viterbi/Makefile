.PHONY: clean

CONDA_ENV:=$(CONDA_PREFIX)

all: libbeam_cuda.so

libbeam_cuda.so: viterbi_1.o viterbi.o
	nvcc $(DEBUG) viterbi_1.o viterbi.o -o libbeam_cuda.so -Xcompiler \"-Wl,-rpath,$(CONDA_ENV)/lib\" --library-path=$(CONDA_ENV)/lib --library=python3.10 --shared

%.o:%.cpp
	nvcc $(DEBUG) -dc -x cu  -c --include-path $(CONDA_PREFIX)/include/python3.10/ --include-path $(CONDA_PREFIX)/include/ --extended-lambda --compiler-options '-fPIC' -o $@ --std=c++14 $<

%.o:%.cu
	nvcc $(DEBUG) -dc -Xptxas='-v' -c -x cu --include-path $(CONDA_PREFIX)/include/python3.10/ --include-path $(CONDA_PREFIX)/include/ --extended-lambda --compiler-options '-fPIC' --std=c++14 -o $@ $<

clean:
	rm libbeam_cuda.so *.o

