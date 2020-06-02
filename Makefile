CXX= /usr/local/bin/g++-8 #g++
CXX_FLAGS= -std=c++11 -fopenmp -O2

# Include and Library directories
INC= -I/usr/include -I./include
LIB= -lgomp

# File directory structure
SRC_DIR= src
OBJ_DIR= obj
BIN_DIR= bin

CPP_FILES= $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES= $(addprefix $(OBJ_DIR)/,$(notdir $(CPP_FILES:.cpp=.o)))

# Outputs
OBJS= $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir $(CPP_FILES)))
EXEC= $(addprefix $(BIN_DIR)/, imgconvert)

# Make directories
mkdirs:= $(shell mkdir -p $(OBJ_DIR) $(BIN_DIR))


# Build everything
all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) -o $@ $^ $(LIB)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXX_FLAGS) $(INC) -c -o $@ $<


# Remove old files
clean:
	rm -f $(OBJS) $(EXEC)

