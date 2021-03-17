# Check for OS (Windows, Linux, Mac OS)
ifeq ($(OS),Windows_NT)
	DETECTED_OS:= Windows
else
	DETECTED_OS:= $(shell uname)
endif

# Set compiler and flags
CXX= g++
CXXFLAGS+= -std=c++11 -O2

# Set source and output directories
ifeq ($(DETECTED_OS),Windows)
	TESTSRCDIR= test\src
	TESTOBJDIR= test\obj
	TESTBINDIR= test\bin
else
	TESTSRCDIR= test/src
	TESTOBJDIR= test/obj
	TESTBINDIR= test/bin
endif

# Set up include and libray directories
ifeq ($(DETECTED_OS),Windows)
	TESTINC= -I"$(HOMEPATH)\local\include" -I.\include
	TESTLIB= -L"$(HOMEPATH)\local\lib" -L.\lib -lglfw3dll -lparicompress
else 
	TESTINC= -I$(HOME)/local/include -I./include
	TESTLIB= -L$(HOME)/local/lib -lGL -lglfw -lcudart
endif

# Create output directories and set output file names
ifeq ($(DETECTED_OS),Windows)
	mkobjdir:= $(shell if not exist $(TESTOBJDIR) mkdir $(TESTOBJDIR))
	mkbindir:= $(shell if not exist $(TESTBINDIR) mkdir $(TESTBINDIR))

	TESTOBJS= $(addprefix $(TESTOBJDIR)\, sample.o)
	TESTEXEC= $(addprefix $(TESTBINDIR)\, sample.exe)
else
	mkdirs:= $(shell mkdir -p $(TESTOBJDIR) $(TESTBINDIR))

	TESTOBJS= $(addprefix $(TESTOBJDIR)/, sample.o)
	TESTEXEC= $(addprefix $(TESTBINDIR)/, sample)
endif


# BUILD EVERYTHING# BUILD EVERYTHING
all: todo

todo:
	echo "TODO: implement compile library on Linux"

test: $(TESTEXEC)
$(TESTEXEC): $(TESTOBJS)
	$(CXX) -o $@ $^ $(TESTLIB)

ifeq ($(DETECTED_OS),Windows)
$(TESTOBJDIR)\\%.o: $(TESTSRCDIR)\%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $< $(TESTINC)
else
$(TESTOBJDIR)/%.o: $(TESTSRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $< $(TESTINC)
endif

# REMOVE OLD FILES
ifeq ($(DETECTED_OS),Windows)
clean:
	del $(TESTOBJS) $(TESTEXEC)
else
clean:
	rm -f $(TESTOBJS) $(TESTEXEC)
endif
