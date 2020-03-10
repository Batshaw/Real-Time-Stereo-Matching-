#include "Application.hpp"

//callback functions for radio buttons

/*
void Application::on_button_SAD_green_similarity(int state, void* userdata) {
    active_stereo_matching_mode = dsm::StereoMatchingMode::SIMPLE_SAD_GREEN_3x8;
    m_changed_matching_state = true;
}

void Application::on_button_SAD_grayscale_similarity(int state, void* userdata) {
    active_stereo_matching_mode = dsm::StereoMatchingMode::SIMPLE_CENSUS_GRAYSCALE_1x8;
    m_changed_matching_state = true;
}

void Application::on_button_SAD_grayscale_local_memory_similarity(int state, void* userdata) {
    active_stereo_matching_mode = dsm::StereoMatchingMode::SIMPLE_SAD_GRAYSCALE_LOCAL_MEMORY_1x8;
    m_changed_matching_state = true;
}

void Application::on_button_SAD_lab_32f_similarity(int state, void* userdata) {
    active_stereo_matching_mode = dsm::StereoMatchingMode::SIMPLE_SAD_LAB_3x32F;
    m_changed_matching_state = true;
}

void Application::on_button_ASW_lab_similarity(int state, void* userdata) {
    active_stereo_matching_mode = dsm::StereoMatchingMode::SIMPLE_ASW_LAB_3x32F;
    m_changed_matching_state = true;
}

*/


// callback functions for sliders
void Application::on_search_window_size_trackbar(int state, void* userdata) {
    m_changed_matching_state = true;
}

void Application::on_minimum_disparity_trackbar(int state, void* userdata) {
    m_changed_matching_state = true;
}

void Application::on_maximum_disparity_trackbar(int state, void* userdata) {
    m_changed_matching_state = true;
}

void Application::on_disparity_vis_scaling_trackbar(int state, void* userdata) {
  m_disparity_vis_scaling = std::max(1, m_disparity_vis_scaling);
  std::cout << "Changed disparity vis scaling to " << m_disparity_vis_scaling << std::endl;
}


//void Application::on_focal_length_trackbar(int state, void* userdata) {
//  m_changed_matching_state = true;
//}

void Application::on_distance_to_cam_trackbar(int state, void* userdata) {
}

void Application::on_pm_iterations_trackbar(int state, void* userdata) {
  m_changed_matching_state = true;
  
}

void Application::on_pm_temporal_propagation(int state, void* userdata){
    m_changed_matching_state = true;
}

void Application::on_pm_view_propagation(int state, void* userdata){
    m_changed_matching_state = true;
}

void Application::on_pm_plane_refine_steps(int state, void* userdata) {
    m_changed_matching_state = true;
}

void Application::on_pm_slanted_or_fronto(int state, void* userdata) {
    m_changed_matching_state = true;
}

void Application::on_gain_offset_trackbar(int state, void* userdata) {
    stereo_camera_ptr_->set_gain(1.0 + (stereo_rig_gain_offset / 10.0f) );
}



// GLFW GUI CALLBACKS

void Application::glfw_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
        m_accumulate_time = !m_accumulate_time;
    }
}


