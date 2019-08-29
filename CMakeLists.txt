PROJECT(OTColorTransfer)
cmake_minimum_required(VERSION 2.8)


set (CMAKE_CXX_STANDARD 11)

#find openmp
find_package(OpenMP REQUIRED)


# find OpenCL
find_package(OpenCL REQUIRED)
include_directories(SYSTEM ${OpenCL_INCLUDE_DIRS})

# find Boost
find_package(Boost 1.48 REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

set(EXAMPLES
  colorTransfer
  colorTransferGPUv0
  colorTransferGPU
  devices
)

foreach(EXAMPLE ${EXAMPLES})
    add_executable(${EXAMPLE} ${EXAMPLE}.cpp)
    target_link_libraries(${EXAMPLE} PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(${EXAMPLE} ${OpenCL_LIBRARIES} ${Boost_LIBRARIES})
    if(UNIX)
      target_link_libraries(${EXAMPLE} -lpthread -lm)
    endif()
endforeach()
