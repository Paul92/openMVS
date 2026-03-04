# Vendored source lives in this overlay port; no network fetch required.
set(SOURCE_PATH "${CMAKE_CURRENT_LIST_DIR}/src")
if(NOT EXISTS "${SOURCE_PATH}")
    message(FATAL_ERROR
        "mapmap-cpu: missing vendored source directory: '${SOURCE_PATH}'")
endif()

set(_mapmap_cpu_mapmap_dir "")
if(EXISTS "${SOURCE_PATH}/mapmap/full.h")
    set(_mapmap_cpu_mapmap_dir "${SOURCE_PATH}/mapmap")
elseif(EXISTS "${SOURCE_PATH}/full.h")
    set(_mapmap_cpu_mapmap_dir "${SOURCE_PATH}")
else()
    message(FATAL_ERROR
        "mapmap-cpu: could not find mapmap headers in '${SOURCE_PATH}'. "
        "Expected either mapmap/full.h or full.h")
endif()

set(_mapmap_cpu_ext_dir "")
if(EXISTS "${SOURCE_PATH}/ext")
    set(_mapmap_cpu_ext_dir "${SOURCE_PATH}/ext")
elseif(EXISTS "${_mapmap_cpu_mapmap_dir}/ext")
    set(_mapmap_cpu_ext_dir "${_mapmap_cpu_mapmap_dir}/ext")
else()
    message(FATAL_ERROR
        "mapmap-cpu: could not find ext directory in '${SOURCE_PATH}'")
endif()

# Install in the standard include root; consumers can include <mapmap/full.h>.
set(_mapmap_cpu_include_root "${CURRENT_PACKAGES_DIR}/include")
execute_process(COMMAND "${CMAKE_COMMAND}" -E make_directory "${_mapmap_cpu_include_root}/mapmap")
execute_process(COMMAND "${CMAKE_COMMAND}" -E make_directory "${_mapmap_cpu_include_root}/mapmap/ext")
execute_process(COMMAND "${CMAKE_COMMAND}" -E copy_directory
    "${_mapmap_cpu_mapmap_dir}"
    "${_mapmap_cpu_include_root}/mapmap")
execute_process(COMMAND "${CMAKE_COMMAND}" -E copy_directory
    "${_mapmap_cpu_ext_dir}"
    "${_mapmap_cpu_include_root}/mapmap/ext")

# Remove empty helper directories so post-build validation does not report
# non-portable empty install folders.
file(GLOB_RECURSE _mapmap_cpu_installed_entries
    LIST_DIRECTORIES true
    "${_mapmap_cpu_include_root}/*")
list(SORT _mapmap_cpu_installed_entries)
list(REVERSE _mapmap_cpu_installed_entries)
foreach(_entry IN LISTS _mapmap_cpu_installed_entries)
    if(IS_DIRECTORY "${_entry}")
        file(GLOB _entry_children "${_entry}/*")
        if("${_entry_children}" STREQUAL "")
            file(REMOVE_RECURSE "${_entry}")
        endif()
    endif()
endforeach()

file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/mapmap-cpu-config.cmake"
    DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}"
    RENAME "mapmap-cpu-config.cmake")
file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/mapmap-cpu-config.cmake"
    DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}"
    RENAME "mapmap-cpuConfig.cmake")

file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/usage"
    DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")

set(_mapmap_cpu_license "")
foreach(_candidate LICENSE.txt LICENSE LICENSE.md)
    if(EXISTS "${SOURCE_PATH}/${_candidate}")
        set(_mapmap_cpu_license "${SOURCE_PATH}/${_candidate}")
        break()
    endif()
endforeach()

if(_mapmap_cpu_license STREQUAL "")
    message(FATAL_ERROR
        "mapmap-cpu: no license file found in ${SOURCE_PATH}")
endif()

vcpkg_install_copyright(FILE_LIST "${_mapmap_cpu_license}")
