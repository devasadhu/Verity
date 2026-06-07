CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -fPIC -std=c99
TARGET  = hashmap.so

$(TARGET): hashmap.c hashmap.h
	$(CC) $(CFLAGS) -shared -o $(TARGET) hashmap.c

clean:
	rm -f $(TARGET) *.o

.PHONY: clean