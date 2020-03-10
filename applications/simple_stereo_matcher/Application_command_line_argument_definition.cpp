#include "Application.hpp"
//define command line arguments
void Application::define_command_line_arguments() {
    desc.add_options() 
      ("help,h", "Print help messages") //bool
      ("image_1,i", po::value<std::string>(), "Path to image 1") //std::string
      ("image_2,j", po::value<std::string>(), "Path to image 2") //std::string
      ("camera,c", po::value<std::string>(), "path to camera parameters: should be like: applications/images/Checkerboard_Baumer/camera_params.yml") //std::string
      ("reference,r", po::value<std::string>(), "Path to reference disparity 1 to 2") //std::string
      ("no-gui,n", "No GUI") //bool
      ("output_path,o", po::value<std::string>(), "Path to output point ckoud (needs to end in .ply)") //std::string
      ("win_size,w", po::value<std::vector<int>>()->multitoken(), "2D size of window in gui x y") //2 ints
      ("use_adcensus,a", "If this flag is set, the adcensus matcher is used")
      ("use_patch_match,p", po::value<std::vector<int>>()->multitoken(), "patch match with user-defined num_iteration , min_disp , max_disp")
      ("use_stereo_camera,s", "If this flag is set, a baumer camera array is initialized and used.")
      ("tcp_socket_out,t",  po::value<std::string>(), "zmq bind socket address to send reconstruction data from (server_ip:port)") //std::string
      
      ("input-initial-guess-3,g", po::value<std::string>(),  "initial guess image input - Path to image 3") //bool
      ("input-initial-guess-4,u", po::value<std::string>(),  "initial guess image input - Path to image 4") //bool

      ("downsampling_factor,d", po::value<float>(), "downsample image along each axis by this if d < 1.0") 

      ("color,x", "run camera in color mode (only makes sense if this is supported by the camera)") 
      ;
}
