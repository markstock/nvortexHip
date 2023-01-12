#
# nvortexHip
#
PROJECT (nvortexHip)
CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
IF (NOT CMAKE_BUILD_TYPE)
  SET (CMAKE_BUILD_TYPE "Release")
ENDIF ()
SET (CMAKE_BUILD_TYPE "Release" CACHE STRING "Select which configuration to build")
#SET (USE_OMP FALSE CACHE BOOL "Use OpenMP multithreading")
#SET (USE_VC FALSE CACHE BOOL "Use Vc for vector arithmetic")
SET (CMAKE_VERBOSE_MAKEFILE on)
SET (CMAKE_EXPORT_COMPILE_COMMANDS on)
IF (NOT CMAKE_BUILD_TYPE)
  SET (CMAKE_BUILD_TYPE "Release")
ENDIF ()

IF (CMAKE_COMPILER_IS_GNUCXX)
  SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wformat -std=c++14")
  #SET (CMAKE_CXX_FLAGS_DEBUG "-O0 -g -ggdb")
  SET (CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
  SET (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -march=native -g -ggdb3")
ELSEIF (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # The Cray compiler reports as Clang to most versions of CMake
  EXECUTE_PROCESS(COMMAND ${CMAKE_CXX_COMPILER} --version
                  COMMAND grep -c Cray
                  OUTPUT_VARIABLE INTERNAL_HAVE_CRAY_COMPILER
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  # and hipcc reports Clang
  EXECUTE_PROCESS(COMMAND ${CMAKE_CXX_COMPILER} --version
                  COMMAND grep -c HIP
                  OUTPUT_VARIABLE INTERNAL_HAVE_HIPCC_COMPILER
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wformat -std=c++14")
  IF (INTERNAL_HAVE_CRAY_COMPILER) # Cray's clang
    # cray compiler misidentifies as Clang
  ELSEIF (INTERNAL_HAVE_HIPCC_COMPILER) # AMD's hipcc
  ELSE() # original clang needs an extra
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
  ENDIF()
  #SET (CMAKE_CXX_FLAGS_DEBUG "-O0 -g -ggdb")
  SET (CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
  SET (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -march=native -g -ggdb")
ELSEIF (MSVC)
  #SET (CMAKE_CXX_FLAGS "/Ox")
ENDIF ()

INCLUDE_DIRECTORIES ( "src" )

ADD_EXECUTABLE ( "nvHip01.bin" "src/nvHip01.cpp" )
ADD_EXECUTABLE ( "nvHip02.bin" "src/nvHip02.cpp" )
ADD_EXECUTABLE ( "nvHip03.bin" "src/nvHip03.cpp" )
ADD_EXECUTABLE ( "nvHip04.bin" "src/nvHip04.cpp" )
ADD_EXECUTABLE ( "nvHip05.bin" "src/nvHip05.cpp" )
ADD_EXECUTABLE ( "ngHip05.bin" "src/ngHip05.cpp" )
ADD_EXECUTABLE ( "ngHip06.bin" "src/ngHip06.cpp" )
ADD_EXECUTABLE ( "ngHip07.bin" "src/ngHip07.cpp" )

