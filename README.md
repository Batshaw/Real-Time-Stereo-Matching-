#Framework for the Real-Time Stereo Matching 2019 project at Bauhaus-Universität Weimar

## Building the project
The project was built under Linux using CMake. Dependencies are OpenCV and OpenCL and Qt5, so make sure that you have them installed.

In the root folder, do the following:

$ mkdir build

$ cd build

$ ccmake ..                     #or alternatively use cmake-gui .. instead of ccmake ..

In cmake, press configure until no CMake-Errors occur anymore (ignore the warnings), then press generate.

If everything went well, you should find a Makefile in the build-Folder.
Continue in the build-Folder:

$ make -j4 install

If everything compiled properly, you will get all executables and dependencies copied into the folder install/bin in your project root.

The dependencies are installed on the lab PCs, such that you have a working environment there for sure.

If you want to work on your own machine as well, you most will need to install a compatible OpenCV version as well as OpenCL headers and drivers yourself. Since we are using the OpenCV-GUI in combination with Qt, you will also need to install Qt5.

Under linux, you should be able to resolve most of your build problems using a package manager like apt-get.

If you have problems building the project yourself, sit together with on of your team mates, us or drop us a mail.

After resolving the dependencies, the project is build like similar cmake projects:




## Project Structure
The project is coarsely structured into three components:
  
  - framework: contains C++ and C source code which can be reused by any application (like ImageProcessing classes doing the low level CL work in a more structured way)
  - application: actual applications which will be compiled into executables (but you need to tell CMake that you want to compile this, see next section)
  - kernels: Source-Code for OpenCL Kernels (ending on *.cl), which are coarsely structured based on their task. cl-files can include other cl-files (similar to C++-Code using include. For an example, have a short look into the the kernel in kernels/stereo_matching/matching_algorithms/simple_SAD_grayscale_1x8.cl which includes the cost function kerlen file SAD_grayscale_1x8.cl)

 ## Adding new Applications
 You can rather easily add new source files by copying existing ones in the folder "applcations", cleaning them up according to your needs and renaming them appropriately. However, you need to tell CMake, that you want to compile the new app.
 
 For this, open the CMakeLists.txt file in the folder applications/ (not the one in the root project directory)

 Here, you find entries like:

\#image_blender

add_executable(image_blender image_blender/main.cpp)

target_link_libraries(image_blender dense_stereo_matching ${Boost_LIBRARIES})

target_include_directories(image_blender PUBLIC ${OpenCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} )

install(TARGETS image_blender DESTINATION bin)

Copy such a block and replace every instance of the application name (here: image_blender) by how your compiled application should be called. Furthermore, replace the path to your own main.cpp file in the add_executables call.

Save the CMakeLists file, and run cmake .. again in the build folder. If everything worked, cmake should encounter no problems and you can build the project including your own app!

If CMake encountered an error, the error messages (or your colleagues) usually help you out quite well.


# Important Links
CL-Error codes (what is stored in cl-status, helps you out quite well sometimes!):
	https://streamhpc.com/blog/2013-04-28/opencl-error-codes/

OpenCL 2.2 Specs:
	https://www.khronos.org/registry/OpenCL/specs/2.2/pdf/OpenCL_C.pdf

OpenCL C-Language Specification:
	https://www.khronos.org/registry/OpenCL/specs/2.2/html/OpenCL_C.html#the-opencl-c-programming-language

www.search-engine-of-your-choice.com (seriously, often describing your problem well or pasting parts of a error messages helps you understand fastest about whats going on)

Timing (CPU!) execution time using std::chrono: 
	http://www.cplusplus.com/reference/chrono/high_resolution_clock/now/


## Lower Priority Links:
OpenCL-Reference Card (nice as a quick overview, but usually not so helpful when you need to figure out why your kernel is crashing!):
	https://www.khronos.org/files/opencl22-reference-guide.pdf



#3d_recon_app example call:
./3d_recon_app  ./images/Teddy/disp6.png ./images/Teddy/im6.png x.ply 60.0 0.50 4.0

#simple_stereo_matching- PATCH MATCH call:
./simple_stereo_matcher -p <num_iterations> <min_disp> <max_disp> <temporal_prop (1 == OFF)> <change_outlier_detection 0 or 1>
./simple_stereo_matcher -p 4 0 60 1 0 

with initial guess input:
./simple_stereo_matcher -p 4 0 60 1 0 -g ./images/Teddy/disp2_n.png
