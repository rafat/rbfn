include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(SOURCE_FILES    			kmeans.c
					lls.c
					matrix.c
					netdata.c
					rbf.c
                    )

set(HEADER_FILES    			kmeans.h
					lls.h
					matrix.h
					netdata.h
					rbf.h
                    )

add_library(rbfnlib STATIC ${SOURCE_FILES} ${HEADER_FILES})
        
set_property(TARGET rbfnlib PROPERTY FOLDER "lib")

target_include_directories(rbfnlib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})