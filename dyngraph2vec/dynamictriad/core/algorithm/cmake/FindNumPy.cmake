# Find the native numpy includes
# This module defines
#  PYTHON_NUMPY_INCLUDE_DIR, where to find numpy/arrayobject.h, etc.
#  PYTHON_NUMPY_FOUND, If false, do not try to use numpy headers.
if (NOT PYTHON_NUMPY_INCLUDE_DIR)
    exec_program ("${PYTHON_EXECUTABLE}"
      ARGS "-c" "\"import numpy; print(numpy.get_include())\""
      OUTPUT_VARIABLE PYTHON_NUMPY_INCLUDE_DIR
      RETURN_VALUE NUMPY_NOT_FOUND)
    if (PYTHON_NUMPY_INCLUDE_DIR MATCHES "Traceback")
    # Did not successfully include numpy
      set(PYTHON_NUMPY_FOUND FALSE)
    else (PYTHON_NUMPY_INCLUDE_DIR MATCHES "Traceback")
    # successful
      set (PYTHON_NUMPY_FOUND TRUE)
      set (PYTHON_NUMPY_INCLUDE_DIR ${PYTHON_NUMPY_INCLUDE_DIR} CACHE PATH "Numpy include path")
    endif (PYTHON_NUMPY_INCLUDE_DIR MATCHES "Traceback")
    if (PYTHON_NUMPY_FOUND)
      if (NOT NUMPY_FIND_QUIETLY)
        message (STATUS "Numpy headers found")
      endif (NOT NUMPY_FIND_QUIETLY)
    else (PYTHON_NUMPY_FOUND)
      if (NUMPY_FIND_REQUIRED)
        message (FATAL_ERROR "Numpy headers missing")
      endif (NUMPY_FIND_REQUIRED)
    endif (PYTHON_NUMPY_FOUND)
    mark_as_advanced (PYTHON_NUMPY_INCLUDE_DIR)
endif (NOT PYTHON_NUMPY_INCLUDE_DIR)

