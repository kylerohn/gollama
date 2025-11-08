add_library(shims SHARED src/shims.cpp)
target_include_directories(shims PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(shims PRIVATE llama)

set_target_properties(shims PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

