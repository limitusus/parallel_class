CC:=/usr/bin/mpicc
CFLAGS=-Wall -O3
TARGETS=clustering

all : $(TARGETS)

clustering : clustering.o
	$(CC) $(CFLAGS) -o $@ $< -lm

clean:
	rm -f clustering clustering.o
