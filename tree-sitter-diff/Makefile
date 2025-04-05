# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra
CFLAGS += -ansi -pedantic -D_DEFAULT_SOURCE # enforce ansi c standards (allowing some stuff ;)
CLFAGS += -g # debug
INCLUDES = -I/usr/local/include
LDFLAGS = -L/usr/local/lib
LIBS = -ltree-sitter -ltree-sitter-c

# Target executable
TARGET = tsdiff

# Source files
SOURCES = \
	$(wildcard *.c)

# Object files
OBJECTS = $(SOURCES:.c=.o)

# Default target
all: $(TARGET)

# Link the target executable
$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

# Compile source files to object files
%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Clean up build artifacts
clean:
	rm -f $(TARGET) $(OBJECTS)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Phony targets
.PHONY: all clean run
