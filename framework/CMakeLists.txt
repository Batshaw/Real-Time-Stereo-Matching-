file(GLOB_RECURSE FRAMEWORK_SOURCE *.cu *.cpp *.c)
file(GLOB_RECURSE FRAMEWORK_INLINE *.inl)
file(GLOB_RECURSE FRAMEWORK_HEADER *.cuh *.hpp *.h)

source_group("Inline Files" FILES ${FRAMEWORK_INLINE})

link_directories(${BAUMER_LIBRARY_DIR})

#cuda_add_library 
add_library(dense_stereo_matching STATIC
  ${FRAMEWORK_SOURCE}
  ${FRAMEWORK_INLINE}
  ${FRAMEWORK_HEADER})

#self built libraries
#target_link_libraries(dense_stereo_matching globjects glbinding)
#global libraries



target_link_libraries(dense_stereo_matching GLU ${BOOST_LIBRARIES} gmp mpfr -fopenmp -lzmq ${Boost_LIBRARIES}  ${OpenCL_LIBRARIES} ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES} ${BAUMER_LIBRARIES})

set_target_properties(dense_stereo_matching PROPERTIES COMPILE_FLAGS "-fopenmp")



target_include_directories(dense_stereo_matching PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${BAUMER_INCLUDE_DIR})
target_include_directories(dense_stereo_matching PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core)
target_include_directories(dense_stereo_matching PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/framework)
target_include_directories(dense_stereo_matching PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/image_processing)
target_include_directories(dense_stereo_matching PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/protocols)
target_include_directories(dense_stereo_matching PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/libBaumer)
