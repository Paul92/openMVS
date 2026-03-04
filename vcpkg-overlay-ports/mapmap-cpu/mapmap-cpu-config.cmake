if(NOT TARGET mapmap-cpu::mapmap-cpu)
    add_library(mapmap-cpu::mapmap-cpu INTERFACE IMPORTED)
    set_target_properties(mapmap-cpu::mapmap-cpu PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES
            "${CMAKE_CURRENT_LIST_DIR}/../../include"
    )
endif()
