PHONY: install

CONDA_ENV:=$(CONDA_PREFIX)
LIB_INSTALL_DIR:=$(CONDA_ENV)/lib
INCLUDE_INSTALL:=$(CONDA_ENV)/include/

SOURCES:=hedges_hooks_c.cpp fast_hedges.cpp

COMPILER=$(CXX)
override CXXFLAGS += -std=c++11 -O3 -fPIC -isystem $(CONDA_ENV)/include/python3.10

OBJECTS:=hedges_hooks_c.o fast_hedges.o

OUTPUT:=libhedges_hooks_c.so

install: $(OBJECTS)
	cp hedges_hooks_c.h $(INCLUDE_INSTALL)
	$(COMPILER) -shared -o $(OUTPUT) $(OBJECTS) -L$(CONDA_ENV)/lib -lpython3.10 -Wl,-rpath=$(CONDA_ENV)/lib
	cp $(OUTPUT) $(LIB_INSTALL_DIR)/$(OUTPUT)

%.o:%.cpp
	$(COMPILER) $(CXXFLAGS) $(INCLUDE) -c $< -o $@ 


clean:
	rm *.so *.o
	rm $(LIB_INSTALL_DIR)/$(OUTPUT)
	rm $(INCLUDE_INSTALL)/hedges_hooks_c.h
