CUDA_ROOT_DIR=/usr/local/cuda
CXX= g++
CXX_FLAGS= -std=c++11
NVCC  := nvcc -ccbin /usr/bin
INC=-I./include
LIB= 

# NVCC compiler options:
NVCC=nvcc
NVCC_FLAGS=
NVCC_LIBS=

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
CUDA_LINK_LIBS= -lcudart

#file directory:
SRC_DIR = src
BINDIR = bin
OBJ_DIR =  obj
#INC_DIR = include

OBJS = $(addprefix $(OBJ_DIR)/,main.o) 
TARGET = $(addprefix $(BINDIR)/, gray)
mkdirs:= $(shell mkdir -p $(OBJDIR) $(BINDIR))

# Compile CUDA source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu # $(INC_DIR)/%.h
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $< $(INC) 
	#$(NVCC) $(NVCC_FLAGS) -c $<  -o $@

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS) $(INC)
# Clean objects in object directory.
clean:
	$(RM) -f $(OBJS) $(EXEC)
#	$(RM) bin/* *.o

