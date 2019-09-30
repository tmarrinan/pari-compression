CXX= g++
CXX_FLAGS= -std=c++11
NVCC= nvcc
NVCC_FLAGS= -G -g 

# Include and Library directories
INC= -I./include
LIB= -L/usr/local/cuda/lib64 -lcudart

# File directory structure
SRC_DIR= src
OBJ_DIR= obj
BIN_DIR= bin

CPP_FILES= $(wildcard $(SRC_DIR)/*.cpp)
CU_FILES= $(wildcard $(SRC_DIR)/*.cu)
OBJ_FILES= $(addprefix $(OBJ_DIR)/,$(notdir $(CPP_FILES:.cpp=.o)))
CUO_FILES= $(addprefix $(OBJ_DIR)/,$(notdir $(CU_FILES:.cu=.cu.o)))

# Outputs
OBJS= $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir $(CPP_FILES)))
OBJS+= $(patsubst %.cu,$(OBJ_DIR)/%.cu.o,$(notdir $(CU_FILES)))
EXEC= $(addprefix $(BIN_DIR)/, imgconvert)

# Make directories
mkdirs:= $(shell mkdir -p $(OBJ_DIR) $(BIN_DIR))


# Build everything
all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) -o $@ $^ $(LIB)

$(OBJ_DIR)/%.cu.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) $(INC) -c -o $@ $<

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXX_FLAGS) $(INC) -c -o $@ $<


# Remove old files
clean:
	rm -f $(OBJS) $(EXEC)

