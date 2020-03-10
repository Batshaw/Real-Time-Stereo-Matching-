#ifndef DSM_GEOMETRY_SENDER_H
#define DSM_GEOMETRY_SENDER_H

#include <glm/gtc/matrix_transform.hpp>

#include <map>
#include <string>
#include <vector>

#include <atomic>
#include <unordered_map>

#include <sgtp/SGTP.h>

namespace boost {
class thread;
class mutex;
}

namespace dsm {
namespace sys {

using send_header_type  = SGTP::header_data_t;
using send_package_type = SGTP::send_package_t;

class GeometrySender {
  public:
    GeometrySender(const char* server_socket);
    ~GeometrySender();

    //void send_data(send_package_type const& send_package_type, 
    //               int32_t reply_package_id = -1);
    void send_data(send_package_type const& send_package_type, 
                   send_header_type const& additional_header_info = send_header_type() );    


 private:

  void _send_loop(std::string const& server_socket);

  bool m_running;
  boost::thread* m_send_thread;
  boost::mutex* m_mutex;
  boost::mutex* m_submission_mutex;
  //client_feedback m_fb;


  // buffers containing compressed data before they are copied into the message
  std::size_t m_compressed_geometry_data_size = 0;
  std::array<std::size_t, 16> m_compressed_image_sizes_per_layer;
  std::unordered_map<int, std::array<uint8_t, 1280*720*3> > m_compressed_image_cache_per_layer;
  std::array<uint8_t, 1024*1024*50> m_total_compressed_geometry_data;


  std::vector<uint8_t*> m_compressed_tj_image_buffers = std::vector<uint8_t*>(16, nullptr);

  //std::vector<client_feedback> m_all_clients_fb;
  std::atomic<bool> m_new_data_available;

  uint8_t* m_calibration_send_buffer;
  uint64_t m_calibration_to_send_size;

  uint8_t* m_send_buffer;
  uint64_t m_message_to_send_size;
  uint8_t* m_send_buffer_back;
  uint64_t m_message_to_send_size_back;

  float   m_compression_time_milliseconds = 0.0;
  float   m_compression_time_milliseconds_back = 0.0;


  std::atomic<bool>     m_is_new_message;

  bool     m_apply_jpeg_compression_back;
  bool     m_apply_jpeg_compression;
  int32_t  m_jpeg_quality;
  
  std::size_t const MAX_SEND_BUFFER_SIZE;
};
} //namespace sys
} //namespace dsm

#endif