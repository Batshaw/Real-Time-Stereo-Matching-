link_directories(${BAUMER_LIBRARY_DIR})


####### SIMPLE EXAMPLES START #############

#image differences using buffers 
set(APP_NAME simple_example_compute_image_differences_using_buffers)
add_executable(${APP_NAME} simple_examples/compute_image_differences_using_buffers.cpp)
target_link_libraries(${APP_NAME} dense_stereo_matching ${Boost_LIBRARIES})
target_include_directories(${APP_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} )
install(TARGETS ${APP_NAME} DESTINATION bin)

set(APP_NAME simple_example_window_based_image_binarization)
add_executable(${APP_NAME} simple_examples/window_based_image_binarization.cpp)
target_link_libraries(${APP_NAME} dense_stereo_matching ${Boost_LIBRARIES})
target_include_directories(${APP_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} )
install(TARGETS ${APP_NAME} DESTINATION bin)

#sobel app
set(APP_NAME sobel_filter)
add_executable(${APP_NAME} simple_tasks/sobel_filter.cpp)
target_link_libraries(${APP_NAME} dense_stereo_matching ${Boost_LIBRARIES})
target_include_directories(${APP_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} )
install(TARGETS ${APP_NAME} DESTINATION bin)

#gaussian filter app
set(APP_NAME gaussian_filter)
add_executable(${APP_NAME} simple_tasks/gaussion_filter.cpp)
target_link_libraries(${APP_NAME} dense_stereo_matching ${Boost_LIBRARIES})
target_include_directories(${APP_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} )
install(TARGETS ${APP_NAME} DESTINATION bin)

#simple_patch_match app
set(APP_NAME simple_patch_match)
add_executable(${APP_NAME} simple_tasks/patch_match.cpp)
target_link_libraries(${APP_NAME} dense_stereo_matching ${Boost_LIBRARIES})
target_include_directories(${APP_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} )
install(TARGETS ${APP_NAME} DESTINATION bin)

####### SIMPLE EXAMPLES END #############




#image_converter
set(APP_NAME image_converter)
add_executable(${APP_NAME} image_converter/main.cpp)
target_link_libraries(${APP_NAME} dense_stereo_matching ${Boost_LIBRARIES})
target_include_directories(${APP_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} )
install(TARGETS ${APP_NAME} DESTINATION bin)

#image_blender
set(APP_NAME image_blender)
add_executable(${APP_NAME} image_blender/main.cpp)
target_link_libraries(${APP_NAME} dense_stereo_matching ${Boost_LIBRARIES})
target_include_directories(${APP_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} )
install(TARGETS ${APP_NAME} DESTINATION bin)

#simple_stereo_matcher
set(APP_NAME simple_stereo_matcher)
add_executable(${APP_NAME} simple_stereo_matcher/main.cpp 
						   simple_stereo_matcher/Application.cpp
						   simple_stereo_matcher/Application_command_line_argument_definition.cpp
						   simple_stereo_matcher/Application_gui_definition.cpp
						   simple_stereo_matcher/Application_gui_callback_functions.cpp
						   simple_stereo_matcher/Application_visualization.cpp)
target_link_libraries(${APP_NAME} dense_stereo_matching GL GLU glfw ${Boost_LIBRARIES})
target_include_directories(${APP_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} )
install(TARGETS ${APP_NAME} DESTINATION bin)

#const win stereo_matcher
set(APP_NAME const_win_stereo_matcher)
file(GLOB my_sources "const_win_stereo_matcher/*.h" "const_win_stereo_matcher/*.cpp")
add_executable(${APP_NAME} ${my_sources})
target_link_libraries(${APP_NAME} dense_stereo_matching ${Boost_LIBRARIES})
target_include_directories(${APP_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} )
install(TARGETS ${APP_NAME} DESTINATION bin)

#adaptive weight stereo_matcher
set(APP_NAME adaptive_weight_stereo_matcher)
file(GLOB my_sources "adaptive_weight_stereo_matcher/*.h" "adaptive_weight_stereo_matcher/*.cpp")
add_executable(${APP_NAME} ${my_sources})
target_link_libraries(${APP_NAME} dense_stereo_matching ${Boost_LIBRARIES})
target_include_directories(${APP_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} )
install(TARGETS ${APP_NAME} DESTINATION bin)

#rectify stereo chessboard app
set(APP_NAME rectify_chessboard)
file(GLOB my_sources "rectify_chessboard/*.h" "rectify_chessboard/*.cpp")
add_executable(${APP_NAME} ${my_sources})
target_link_libraries(${APP_NAME} dense_stereo_matching ${Boost_LIBRARIES})
target_include_directories(${APP_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} )
install(TARGETS ${APP_NAME} DESTINATION bin)

# gaussian blur kernel experiment
set(APP_NAME experiments_gaussian_blur)
add_executable(${APP_NAME} experiments/gaussian_blur.cpp)
target_link_libraries(${APP_NAME} dense_stereo_matching ${Boost_LIBRARIES})
target_include_directories(${APP_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} )
install(TARGETS ${APP_NAME} DESTINATION bin)

	
#rectify stereo chessboard app
set(APP_NAME stereo_rig_streamer)
file(GLOB stereo_rig_streamer_sources "stereo_rig_streamer/*.cpp")
add_executable(${APP_NAME} ${stereo_rig_streamer_sources})

target_link_libraries(${APP_NAME} dense_stereo_matching ${Boost_LIBRARIES} ${BAUMER_LIBRARIES})
target_include_directories(${APP_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${BAUMER_INCLUDE_DIR} )
install(TARGETS ${APP_NAME} DESTINATION bin)


#set(APP_NAME opencv_stereo_calib)
#file(GLOB my_sources "opencv_stereo_calib/*.h" "opencv_stereo_calib/*.cpp")
#add_executable(${APP_NAME} ${my_sources})
#target_link_libraries(${APP_NAME} dense_stereo_matching ${Boost_LIBRARIES} ${BAUMER_LIBRARIES})
#target_include_directories(${APP_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${BAUMER_INCLUDE_DIR})
#install(TARGETS ${APP_NAME} DESTINATION bin)

#image difference using image2d app
set(APP_NAME compute_image_differences_using_image_2d)
add_executable(${APP_NAME} non_reworked_examples/compute_image_differences_using_image_2d.cpp)
target_link_libraries(${APP_NAME} dense_stereo_matching )
target_include_directories(${APP_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS})
install(TARGETS ${APP_NAME} DESTINATION bin)

#patch match stereo app
set(APP_NAME patch_match_stereo)
add_executable(${APP_NAME} patch_match_stereo.cpp)
target_link_libraries(${APP_NAME} dense_stereo_matching)
target_include_directories(${APP_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS})
install(TARGETS ${APP_NAME} DESTINATION bin)



#gaussion filter app
#set(APP_NAME gaussion_filter)
#add_executable(${APP_NAME} simple_tasks/gaussion_filter.cpp)
#target_link_libraries(${APP_NAME} dense_stereo_matching)
#target_include_directories(${APP_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS})
#install(TARGETS ${APP_NAME} DESTINATION bin)



#cv gui test app
set(APP_NAME cv_gui_test)
add_executable(${APP_NAME} cv_gui_test.cpp)
target_link_libraries(${APP_NAME} dense_stereo_matching)
target_include_directories(${APP_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
install(TARGETS ${APP_NAME} DESTINATION bin)



set(APP_NAME 3d_recon_app)
#set(APP_NAME ${APP_NAME})
add_executable(${APP_NAME} reconstruct_3D_point_cloud.cpp)
target_link_libraries(${APP_NAME} dense_stereo_matching)
install(TARGETS ${APP_NAME} DESTINATION bin)

set(APP_NAME 3d_gpu_app)
#set(APP_NAME ${APP_NAME})
add_executable(${APP_NAME} simple_tasks/reconstruct_3D_points_GPU.cpp)
target_link_libraries(${APP_NAME} dense_stereo_matching)


#simple_stereo_matcher
#set(APP_NAME pipeline_stereo_matcher)
#add_executable(${APP_NAME} pipeline_stereo_matcher/main.cpp)
#target_link_libraries(${APP_NAME} dense_stereo_matching ${Boost_LIBRARIES})
#target_include_directories(${APP_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} )
#install(TARGETS ${APP_NAME} DESTINATION bin)
