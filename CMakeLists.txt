PROJECT(OTColorTransfer)
cmake_minimum_required(VERSION 2.8)

set (CMAKE_CXX_STANDARD 11)

set(EXAMPLES
  colorTransfer
)

foreach(EXAMPLE ${EXAMPLES})
    add_executable(${EXAMPLE} ${EXAMPLE}.cpp)
    if(UNIX)
      target_link_libraries(${EXAMPLE} -lpthread -lm)
    endif()
endforeach()
