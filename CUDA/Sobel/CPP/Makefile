
CC        := g++ -c
LINK      := g++ -fPIC

all: sobel_edge.o
	$(LINK) -o sobel sobel_edge.o 


complie: sobel_edge.c
	$(CC) sobel_edge.c -o sobel_edge.o

clean:
	rm sobel_edge.o sobel lena_sobel.bmp *~
