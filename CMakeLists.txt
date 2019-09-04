PROJECT(OTColorTransfer)
cmake_minimum_required(VERSION 2.8)

set (CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -fno-signed-zeros -fno-trapping-math -funroll-loops")

#find openmp
find_package(OpenMP REQUIRED)
IF(APPLE)
  include_directories("/usr/local/opt/libomp/include/")
ENDIF()
include_directories(${PROJECT_SOURCE_DIR})

add_library(spot STATIC UnbalancedSliced/UnbalancedSliced.cpp)
target_link_libraries(spot PUBLIC OpenMP::OpenMP_CXX)


set(EXAMPLES
  colorTransfer
  colorTransferPartial
)

foreach(EXAMPLE ${EXAMPLES})
    add_executable(${EXAMPLE} ${EXAMPLE}.cpp)
    target_link_libraries(${EXAMPLE} spot)
    target_link_libraries(${EXAMPLE} OpenMP::OpenMP_CXX)
    if(UNIX)
      target_link_libraries(${EXAMPLE} -lpthread -lm)
    endif()
endforeach()
