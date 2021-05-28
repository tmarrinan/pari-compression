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

# Install directory
INSTALLDIR= $(HOME)/local

# Set source and output directories
ifeq ($(DETECTED_OS),Windows)
	TESTSRCDIR= test\src
	TESTOBJDIR= test\obj
	TESTBINDIR= test\bin
else
	INCDIR= include
	SRCDIR= src
	OBJDIR= obj
	LIBDIR= lib
	TESTSRCDIR= test/src
	TESTOBJDIR= test/obj
	TESTBINDIR= test/bin
endif

# Set up include and libray directories
ifeq ($(DETECTED_OS),Windows)
	TESTINC= -I.\include -I"$(HOMEPATH)\local\include"
	TESTLIB= -L.\lib -L"$(HOMEPATH)\local\lib" -lglfw3dll -lopengl32 -lparicompress
else
	INC= -I./include
	LIB= -lGL -lcudart
	TESTINC= -I./include -I$(HOME)/local/include
	TESTLIB= -L./lib -L$(HOME)/local/lib -lGL -lglfw -lparicompress -lcudart
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

	HEADER= $(addprefix $(INCDIR)/, paricompress.h)
    OBJS= $(addprefix $(OBJDIR)/, paricompress.o)
	LIBR= $(addprefix $(LIBDIR)/, libparicompress.a)
	TESTOBJS= $(addprefix $(TESTOBJDIR)/, sample.o)
	TESTEXEC= $(addprefix $(TESTBINDIR)/, sample)
endif


# BUILD EVERYTHING
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

# INSTALL
install:
ifeq ($(DETECTED_OS),Windows)
	echo "PARI Compress Library must be installed using MSVC"
else
	install -d $(INSTALLDIR)/lib
	install -m 644 $(LIBR) $(INSTALLDIR)/lib
	install -d $(INSTALLDIR)/include
	install -m 644 $(HEADER) $(INSTALLDIR)/include
endif

# REMOVE OLD FILES
ifeq ($(DETECTED_OS),Windows)
clean:
	del $(TESTOBJS) $(TESTEXEC)
else
clean:
	rm -f $(OBJS) $(LIBR) $(TESTOBJS) $(TESTEXEC)
endif
