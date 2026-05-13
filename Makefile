CXX = g++
CXXFLAGS = -std=c++20 -Wall -O2 -I/usr/include -I/usr/local/include -Wno-deprecated-declarations -pthread
LDFLAGS = -lcurl -lyaml-cpp -lssl -lcrypto

TARGET = Autonomous_Coding_Agent
SOURCES = Autonomous_Coding_Agent.cpp
OBJECTS = $(SOURCES:.cpp=.o)

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) *.o execution_traceback.log

run: $(TARGET)
	./$(TARGET)
	
