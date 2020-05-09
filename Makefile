CC=gcc
CFLAGS=

all: mpi.c 
	cc -g -Wall -o mpi mpi.c

clean:
	$(RM) mpi