############
CXX= g++
CXX_FLAGS= -std=c++11

INC= -I./include
LIB= 

SRCDIR= src
OBJDIR= obj
BINDIR= bin

OBJS= $(addprefix $(OBJDIR)/, main.o)
EXEC= $(addprefix $(BINDIR)/, TRLE)

mkdirs:= $(shell mkdir -p $(OBJDIR) $(BINDIR))


# BUILD EVERYTHING
all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) -o $@ $^ $(LIB)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXX_FLAGS) -c -o $@ $< $(INC)


# REMOVE OLD FILES
clean:
	rm -f $(OBJS) $(EXEC)
