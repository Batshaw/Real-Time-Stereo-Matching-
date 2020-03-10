#include "Application.hpp"

void Application::visualize_cv_window() {
  // everything in the show_gui braces is purely meant for the visualization of the results
  // this is completely on the cpu side and helps you to verify whether your code is doing
  // what it is supposed to do        
      cv::Mat grayscale_3x8_mat(m_downsampled_image_height, m_downsampled_image_width, CV_8UC3);
      
      //cv::imshow("x",input_image_1_bgr_3x8);

      
      for(int y = 0; y < output_image_grayscale_1x32f.rows; ++y) {
          for(int x = 0; x < output_image_grayscale_1x32f.cols; ++x) {
              output_image_grayscale_1x8.at<uint8_t>(y, x) = output_image_grayscale_1x32f.at<float>(y, x)  * m_disparity_vis_scaling;
          }
      }
      //output_image_grayscale_1x8 = output_image_grayscale_1x32f;
      cv::cvtColor(output_image_grayscale_1x8, grayscale_3x8_mat, cv::COLOR_GRAY2BGR);

      cv::Mat scaled_out_mat_3x8(m_downsampled_image_height, m_downsampled_image_width, CV_8UC3);

      scaled_out_mat_3x8 = grayscale_3x8_mat;// * m_disparity_vis_scaling;

      //if(accumulate_time)
        //cv::medianBlur(scaled_out_mat_3x8,scaled_out_mat_3x8, 3);

      cv::Mat concatenated_result;

      if(m_camera_parameters_available) {
        cv::hconcat(input_image_1_bgr_3x8_rectified, input_image_2_bgr_3x8_rectified, concatenated_result);
      } else {
        cv::hconcat(input_image_1_bgr_3x8, input_image_2_bgr_3x8, concatenated_result);
      }

      cv::hconcat(concatenated_result, scaled_out_mat_3x8, concatenated_result);
      
      if(reference_disparity_image_3x8.dims == 2) {
        cv::hconcat(concatenated_result, reference_disparity_image_3x8, concatenated_result);

        cv::Mat absolute_difference_image;
        cv::absdiff(scaled_out_mat_3x8, reference_disparity_image_3x8, absolute_difference_image);

        

        cv::hconcat(concatenated_result, absolute_difference_image, concatenated_result);

      }



      cv::imshow(m_window_name, concatenated_result);
      
      // waits one millisecond and returns keycodes if a key was pressed
      int keycode = cv::waitKey(1) & 0xFF;

      //escape
      if(27 == keycode) {
          m_show_gui = false;
      }
      // space
      if(32 == keycode) {
          m_accumulate_time = !m_accumulate_time;
      }

      if('p' == keycode || 'P' == keycode) {
        cv::imwrite("cam_left.png", input_image_1_bgr_3x8_rectified);
        cv::imwrite("cam_right.png", input_image_2_bgr_3x8_rectified);
      }

      //r or R 
      if(114 == keycode || 82 == keycode) {
        stereo_matcher_ptr->reload_kernels(context, device_id);
        std::cout << "Reloaded kernels\n";
        m_changed_matching_state = true;
      }



}

void Application::visualize_geometry() {
    float aspect = window_width / (float) window_height;

    glEnable(GL_DEPTH_TEST); 
    glViewport(0, 0, window_width, window_height);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    gluPerspective(60.0f, aspect ,0.1f,100.0f);

    float BASE_TRANSLATION = m_center_of_rotation_distance_cm * 0.01;
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glTranslatef(0.0, 0.0, -distance_to_vcam_cm * 0.01 );//additional_distance_to_virtual_cam);

    glTranslatef(0.0, 0.0, -BASE_TRANSLATION);

    glDisable(GL_CULL_FACE);
    float current_timestamp = glfwGetTime()*20.0f;
    if(m_accumulate_time) {
      if(0.0f == last_recorded_timestamp) {
        last_recorded_timestamp = current_timestamp;
      }

      float time_delta = current_timestamp - last_recorded_timestamp;
      last_recorded_timestamp = current_timestamp;

      accumulated_time += time_delta;
      accumulated_time = fmod(accumulated_time, 360.0f);
    } else {
      last_recorded_timestamp = current_timestamp;
    }


    glRotatef((float) rot_vertical/10.0f, 1.0f, 0.0f, 0.0f);
    glRotatef((float) rot_horizontal/10.0f, 0.0f, 1.0f, 0.0f);

    glRotatef((float) accumulated_time, 0.f, 1.f, 0.f );

    glRotatef((float)180.0f, 0.f, 1.f, 0.f );
    glRotatef((float)   180.0f, 0.f, 0.f, 1.f );
    glScalef((float)   1.0f, 1.0f, 1.0f);
    glTranslatef((float)   0.0f, 0.f, -(BASE_TRANSLATION) );
    //glPointSize(1.8f);
    glBegin(GL_TRIANGLES);

    float         point_pos[3] = {0.0f, 0.0f, 0.0f};
    float         point_col[3] = {0.0f, 0.0f, 0.0f};


    std::cout << point_positions_vec.size()/3 << std::endl;
    for(uint32_t point_idx = 0; point_idx < (point_positions_vec.size()/3); ++point_idx) {
      int vertex_base_offset = point_idx * 3;
      for(int vertex_idx = 0; vertex_idx < 3; ++vertex_idx) {
        point_pos[vertex_idx] = point_positions_vec[vertex_base_offset + vertex_idx];
        point_col[vertex_idx] = point_color_vec[vertex_base_offset + vertex_idx] / 255.0f;
      }
      glColor3f(point_col[0], point_col[1], point_col[2]);
      glVertex3f(point_pos[0], point_pos[1], point_pos[2]);
    }

    glEnd();

    glfwSwapBuffers(window);
    glfwPollEvents();

}
