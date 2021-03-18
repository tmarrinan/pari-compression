# PARallel Image (PARI) Compression
Real-time parallel image compression using CUDA (optionally with OpenGL interop)

### Compilation Instructions
Windows
* The paricompress CUDA library requires using MSVC
    * Open the 'x64 Native Tools Command Prompt for VS 2019'
    * `cd` to pari-compress project directory
    * Type `nmake /f Makefile.Windows`
        * This should create 'obj' and 'lib' directories
        * The .lib and .dll files will be in the 'lib' directory
* The test sample application can be compiled using Mingw-w64
    * Open a standard Windows command prompt
    * `cd` to pari-compress project directory
    * Type `make test`
        * This should create 'obj' and 'bin' directories inside the 'test' directory 
        * The .exe sample application will be in the 'test\bin' directory
    * Note: paricompress.dll will need to be copied from the 'lib' directory into the 'test\bin' directory and sample.exe must be launched from the 'test' directory 
        
Linux
* Both the library and the test sample application can be compiled using a standard Makefile
    * Open a standard Linux Terminal
    * `cd` to pari-compress project directory
    * Type `make`
        * This should create 'obj' and 'lib' directories
        * The .a file will be in the 'lib' directory
    * Type `make test`
        * This should create 'obj' and 'bin' directories inside the 'test' directory 
        * The executable sample application will be in the 'test\bin' directory
    * Note: sample executable must be launched from the 'test' directory
    
*Note: Makefile may need slight editing to specify where dependencies are installed on your machine*
