SOURCES = mpi.cpp
FLAGS = -std=c++17
COMP = mpic++

all: main

main:
	${COMP} $(SOURCES) -o totallynotavirus ${FLAGS}

clean: 
	rm -rf totallynotavirus