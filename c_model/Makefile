CC := g++
CFLAGS := -Wall -Wextra -std=c++17 -Iinclude $(shell pkg-config --cflags alsa)
LDLIBS = $(shell pkg-config --libs alsa) #can also use
 
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

#create build dir
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(BUILD_DIR)/main.o $(BUILD_DIR)/audio_processing.o
	$(CC) $^ -o $@ $(LDLIBS)

# compile rule
main.o: src/main.cpp | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@ 

audio_processing.o: src/audio_processing.cpp | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(BUILD_DIR)/*.o
