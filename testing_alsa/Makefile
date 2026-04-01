CC = gcc
CFLAGS = -Wall -Wextra -g $(shell pkg-config --cflags alsa)
LDLIBS = $(shell pkg-config --libs alsa)

alsaTest: alsaTest.o
	$(CC) alsaTest.o -o alsaTest $(LDLIBS)

alsaTest.o: alsaTest.c
	$(CC) $(CFLAGS) -c alsaTest.c

clean:
	rm -f alsaTest *.o

