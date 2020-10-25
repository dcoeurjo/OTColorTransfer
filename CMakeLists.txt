PROJECT(OTColorTransfer)
<<<<<<< HEAD
cmake_minimum_required(VERSION 3.12)
=======
cmake_minimum_required(VERSION 3.9)
>>>>>>> 8e4646fa93495df73f7bf4fdd394e6f838ed0bba

set (CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -fno-signed-zeros -fno-trapping-math -funroll-loops")

#find openmp
<<<<<<< HEAD
find_package(OpenMP)
IF(APPLE)
  include_directories("/usr/local/opt/libomp/include/")
ENDIF()
=======
if(APPLE)
    message("-- Setting OpenMP flags on MacOs. Assuming `brew install libomp`")
    if(CMAKE_C_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include" CACHE INTERNAL "OpenMP flags for Xcode toolchain.")
        set(OpenMP_C_LIB_NAMES "omp" CACHE INTERNAL "OpenMP lib name for Xcode toolchain.")
        set(OpenMP_omp_LIBRARY "/usr/local/opt/libomp/lib/libomp.dylib" CACHE INTERNAL "OpenMP lib name for Xcode toolchain.")
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include" CACHE INTERNAL "OpenMP flags for Xcode toolchain.")
        set(OpenMP_CXX_LIB_NAMES "omp" CACHE INTERNAL "OpenMP lib name for Xcode toolchain.")
        set(OpenMP_omp_LIBRARY "/usr/local/opt/libomp/lib/libomp.dylib" CACHE INTERNAL "OpenMP lib name for Xcode toolchain.")
    endif()
endif()
find_package(OpenMP REQUIRED)


>>>>>>> 8e4646fa93495df73f7bf4fdd394e6f838ed0bba
include_directories(${PROJECT_SOURCE_DIR})

add_library(spot STATIC UnbalancedSliced/UnbalancedSliced.cpp)
target_link_libraries(spot PUBLIC OpenMP::OpenMP_CXX)


set(EXAMPLES
  colorTransfer
  colorTransferPartial
  ndTransfer
)

foreach(EXAMPLE ${EXAMPLES})
    add_executable(${EXAMPLE} ${EXAMPLE}.cpp)
    target_link_libraries(${EXAMPLE} spot)
    target_link_libraries(${EXAMPLE} OpenMP::OpenMP_CXX)
    if(UNIX)
      target_link_libraries(${EXAMPLE} -lpthread -lm)
    endif()
endforeach()
