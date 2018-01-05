add_executable(rbftest1 rbftest1.c)

target_link_libraries(rbftest1 rbfnlib m)

add_executable(erbftest1 erbftest1.c)

target_link_libraries(erbftest1 rbfnlib m)

set_target_properties(rbftest1 erbftest1
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/test"
)