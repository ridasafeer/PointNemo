CC := g++
CXXFLAGS := -Wall -Wextra -std=c++17 \
	-Iinclude \
	-I./src/simpleini \
	-I./src/matplotlib-cpp \
	$(shell python3-config --includes) \
	-I$(shell python3 -c "import numpy; print(numpy.get_include())") \
	$(shell pkg-config --cflags alsa)

LDFLAGS := $(shell python3-config --embed --ldflags)
LDLIBS := $(shell pkg-config --libs alsa)
 
SRC_DIR := src
BUILD_DIR := build
TARGET := pointnemo

# -------- the final make file for the entire project

# #define references to the obj files
# SRCS := $(wildcard $(SRC_DIR)/*.cpp)
# OBJS := $(SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# #create build dir
# $(BUILD_DIR):
# 	mkdir -p $(BUILD_DIR)

# #compile step
# $(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
# 	$(CC) $(CFLAGS) -c $< -o $@

# #linker step
# $(TARGET): $(OBJS)
# 	$(CC) $^ -o $@ $(LDLIBS)

# #clean rule

# -------- testing audio_processing make with main()

all: $(TARGET)

$(TARGET): $(BUILD_DIR)/main.o $(BUILD_DIR)/audio_processing.o $(BUILD_DIR)/signal_testing.o $(BUILD_DIR)/controller.o $(BUILD_DIR)/dsp.o $(BUILD_DIR)/fxlms.o
	$(CC) $^ -o $@ $(LDLIBS)

# compile rule
build/main.o: src/main.cpp | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@ 

build/audio_processing.o: src/audio_processing.cpp | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

build/signal_testing.o: src/signal_testing.cpp | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

build/controller.o: src/controller.cpp | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

build/dsp.o: src/dsp.cpp | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

build/fxlms.o: src/fxlms.cpp | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

#create build dir
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)


#automates a loopback test with the voice.wav, outtputing to the micOutput.wav
run: $(TARGET)
	./pointnemo

clean:
	rm -f $(TARGET) $(BUILD_DIR)/*.o
