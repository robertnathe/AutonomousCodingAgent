CXX      = g++
CXXFLAGS = -std=c++20 -Wall -O2 -I/usr/include -I/usr/local/include
LDFLAGS  = -lcurl -lyaml-cpp -lcrypto

TARGET   = output
SOURCES  = Advanced_Coding_Agent.cpp
OBJECTS  = $(SOURCES:.cpp=.o)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) *.o execution_traceback.log

run:
	./$(TARGET)
