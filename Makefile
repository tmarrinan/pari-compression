# Check for OS (Windows, Linux, Mac OS)
ifeq ($(OS),Windows_NT)
	DETECTED_OS:= Windows
else
	DETECTED_OS:= $(shell uname)
endif

# Set compiler and flags
CXX= g++
CXXFLAGS+= -std=c++11 -O2
NVCC= nvcc
NVCCFLAGS+= -ccbin /usr/bin -std=c++11 -O2

# Set source and output directories
ifeq ($(DETECTED_OS),Windows)
	TESTSRCDIR= test\src
	TESTOBJDIR= test\obj
	TESTBINDIR= test\bin
else
	SRCDIR= src
	OBJDIR= obj
	LIBDIR= lib
	TESTSRCDIR= test/src
	TESTOBJDIR= test/obj
	TESTBINDIR= test/bin
endif

# Set up include and libray directories
ifeq ($(DETECTED_OS),Windows)
	TESTINC= -I"$(HOMEPATH)\local\include" -I.\include
	TESTLIB= -L"$(HOMEPATH)\local\lib" -L.\lib -lglfw3dll -lopengl32 -lparicompress
else
	INC= -I./include
	LIB= -lGL -lcudart
	TESTINC= -I$(HOME)/local/include -I./include
	TESTLIB= -L$(HOME)/local/lib -L./lib -lGL -lglfw -lparicompress -lcudart
endif

# Create output directories and set output file names
ifeq ($(DETECTED_OS),Windows)
	mkobjdir:= $(shell if not exist $(TESTOBJDIR) mkdir $(TESTOBJDIR))
	mkbindir:= $(shell if not exist $(TESTBINDIR) mkdir $(TESTBINDIR))

	LIBR= error
	TESTOBJS= $(addprefix $(TESTOBJDIR)\, sample.o)
	TESTEXEC= $(addprefix $(TESTBINDIR)\, sample.exe)
else
	mkdirs:= $(shell mkdir -p $(OBJDIR) $(LIBDIR) $(TESTOBJDIR) $(TESTBINDIR))

	OBJS= $(addprefix $(OBJDIR)/, paricompress.o)
	LIBR= $(addprefix $(LIBDIR)/, libparicompress.a)
	TESTOBJS= $(addprefix $(TESTOBJDIR)/, sample.o)
	TESTEXEC= $(addprefix $(TESTBINDIR)/, sample)
endif


# BUILD EVERYTHING# BUILD EVERYTHING
all: $(LIBR)

ifeq ($(DETECTED_OS),Windows)
$(LIBR):
	echo "PARI Compress Library must be built using MSVC"
else
$(LIBR): $(OBJS)
	ar rcs $(LIBR) $(OBJS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c -Xcompiler -fPIC -Xcompiler -static -o $@ $(INC) $<
endif

test: $(TESTEXEC)

ifeq ($(DETECTED_OS),Windows)
$(TESTEXEC): $(TESTOBJS)
	$(CXX) -o $@ $^ $(TESTLIB)

$(TESTOBJDIR)\\%.o: $(TESTSRCDIR)\%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $(TESTINC) $<
else
$(TESTEXEC): $(TESTOBJS)
	$(NVCC) -o $@ $^ $(TESTLIB)

$(TESTOBJDIR)/%.o: $(TESTSRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $(TESTINC) $<
endif

# REMOVE OLD FILES
ifeq ($(DETECTED_OS),Windows)
clean:
	del $(TESTOBJS) $(TESTEXEC)
else
clean:
	rm -f $(OBJS) $(LIBR) $(TESTOBJS) $(TESTEXEC)
endif
