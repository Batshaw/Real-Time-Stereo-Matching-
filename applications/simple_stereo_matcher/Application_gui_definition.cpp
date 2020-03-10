#include "Application.hpp"

void Application::define_GUI(STEREO_METHOD stereo_method_) {
    std::string param_window_name = "Param_Window";
    cv::namedWindow(param_window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(param_window_name, m_window_size_x, m_window_size_y);

    // parameters for trackbars: element name, window name which it is associated with, address of parameter we need to change, limit, callback function name
    cv::createTrackbar( "Search Window Half Size", param_window_name, &m_search_window_half_size, MAX_CONFIGURABLE_WINDOW_SIZE, Application::on_search_window_size_trackbar );
    cv::createTrackbar( "MinimumDisparity", param_window_name, &m_minimum_disparity, 120, on_minimum_disparity_trackbar );
    cv::createTrackbar( "Maximum Disparity", param_window_name, &m_maximum_disparity, MAX_CONFIGURABLE_DISPARITY, on_maximum_disparity_trackbar );

    cv::createTrackbar( "Disparity Vis Scaling", param_window_name, &m_disparity_vis_scaling, MAX_CONFIGURABLE_DISPARITY_VIS_SCALING, on_disparity_vis_scaling_trackbar );

    //cv::createTrackbar( "Focal Length", m_window_name, &m_focal_length, 3000, on_focal_length_trackbar );
    cv::createTrackbar( "Model to Cam Distance (cm)", param_window_name, &distance_to_vcam_cm, 5000, on_pm_iterations_trackbar );

    //cv::createTrackbar( "Focal Length", m_window_name, &m_focal_length, 3000, on_focal_length_trackbar );
    cv::createTrackbar( "Rotation Horizontal (Deg * 10)", param_window_name, &rot_horizontal, 3600, on_pm_iterations_trackbar );
    cv::createTrackbar( "Rotation Vertical (Deg * 10)", param_window_name, &rot_vertical, 3600, on_pm_iterations_trackbar );

    cv::createTrackbar( "Center of Rotation Distance (cm)", param_window_name, &m_center_of_rotation_distance_cm, 300, NULL );

    if(nullptr != Application::stereo_camera_ptr_ ) {
        cv::createTrackbar( "Camera_Gain Offset", param_window_name, &stereo_rig_gain_offset, 100, on_gain_offset_trackbar );
    }

    if (stereo_method_ == STEREO_METHOD::PATCH_MATCH){
        cv::createTrackbar( "PatchMatch Iterations", param_window_name, &m_num_iteration_propagation, 100, on_pm_iterations_trackbar );
        cv::createTrackbar( "PatchMatch Temporal Iteration OFF-ON", param_window_name, &m_num_temporal_propagation, 1, on_pm_temporal_propagation );
        cv::createTrackbar( "PatchMatch View Iteration OFF-ON", param_window_name, &m_switch_view_prop, 1, on_pm_view_propagation );
        cv::createTrackbar( "PatchMatch Plane Refinement Steps", param_window_name, &m_plane_refine_steps, 20, on_pm_plane_refine_steps);
        cv::createTrackbar( "PatchMatch Slanted Support", param_window_name, &m_slanted_or_fronto, 1, on_pm_slanted_or_fronto);
    }
    else if(stereo_method_ == STEREO_METHOD::AD_CENSUS){
        cv::createTrackbar( "AdCensus::lambdaAD", param_window_name, & m_param_adcensus.lambdaAD, 100, on_pm_iterations_trackbar);
        cv::createTrackbar( "AdCensus::lambdaCensus", param_window_name, & m_param_adcensus.lambdaCensus, 100, on_pm_iterations_trackbar);
        cv::createTrackbar( "AdCensus::tau1", param_window_name, & m_param_adcensus.tau1, 100, on_pm_iterations_trackbar);
        cv::createTrackbar( "AdCensus::tau2", param_window_name, & m_param_adcensus.tau2, 100, on_pm_iterations_trackbar);
        cv::createTrackbar( "AdCensus::tauSO", param_window_name, & m_param_adcensus.tauSO, 100, on_pm_iterations_trackbar);
        cv::createTrackbar( "AdCensus::Pi1", param_window_name, & m_param_adcensus.Pi1, 10, on_pm_iterations_trackbar);
        cv::createTrackbar( "AdCensus::Pi2", param_window_name, & m_param_adcensus.Pi2, 10, on_pm_iterations_trackbar);
        cv::createTrackbar( "AdCensus::VotingThreshold", param_window_name, & m_param_adcensus.votingThreshold, 100, on_pm_iterations_trackbar);
        cv::createTrackbar( "AdCensus::maxSearchDepth", param_window_name, & m_param_adcensus.maxSearchDepth, 100, on_pm_iterations_trackbar);
    }


  // parameters for buttons: element name, callback name, NULL, QT-Element Type
    //cv::createButton("SAD_green", on_button_SAD_green_similarity, NULL, cv::QT_RADIOBOX);
    //cv::createButton("SAD_gray", on_button_SAD_grayscale_similarity, NULL, cv::QT_RADIOBOX);


    //cv::createButton("SAD_lab_32f", on_button_SAD_lab_32f_similarity, NULL, cv::QT_RADIOBOX);
    //cv::createButton("ASW_lab", on_button_ASW_lab_similarity, NULL, cv::QT_RADIOBOX);
}
