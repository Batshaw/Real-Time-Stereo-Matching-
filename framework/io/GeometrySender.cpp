#include "GeometrySender.h"


#include <zmq.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/bind.hpp>
#include <cstdlib>
#include <iostream>

#include <chrono>

// GeometrySender
namespace dsm {
namespace sys {
GeometrySender::GeometrySender(const char* server_socket)
    : m_running(true),
      m_send_thread(),
      m_mutex(new boost::mutex),
      m_submission_mutex(new boost::mutex),
      m_send_buffer(nullptr),
      m_send_buffer_back(nullptr),
      m_message_to_send_size(0),
      m_message_to_send_size_back(0),      
      m_calibration_send_buffer(nullptr),
      m_calibration_to_send_size(0),
      m_is_new_message(false),
      m_apply_jpeg_compression_back(false),
      m_apply_jpeg_compression(false),
      m_jpeg_quality(90),
      MAX_SEND_BUFFER_SIZE(SGTP::MAX_MESSAGE_SIZE) {

  if(nullptr == m_send_buffer) {
    m_send_buffer = new uint8_t[MAX_SEND_BUFFER_SIZE];
    m_send_buffer_back = new uint8_t[MAX_SEND_BUFFER_SIZE];
  }

  if(nullptr == m_calibration_send_buffer) {
    m_calibration_send_buffer = new uint8_t[1024*1014*1024];
  }

  std::string const server_s(server_socket);


  m_send_thread = new boost::thread(
      boost::bind(&GeometrySender::_send_loop, this, server_s));
}

GeometrySender::~GeometrySender() {
  m_running = false;
  m_send_thread->join();

  if(nullptr != m_mutex) {
    delete m_mutex;
    m_mutex = nullptr;
  }

  if(nullptr != m_submission_mutex) {
    delete m_submission_mutex;
    m_submission_mutex = nullptr;
  }

  if(nullptr != m_send_buffer) {
    delete[] m_send_buffer;
    m_send_buffer = nullptr;
  }
  if(nullptr != m_send_buffer_back) {
    delete[] m_send_buffer_back;
    m_send_buffer_back = nullptr;
  }


}



void GeometrySender::send_data(send_package_type const& send_package, send_header_type const& additional_header_info) {
  boost::mutex::scoped_lock lock(*m_submission_mutex);


  SGTP::header_data_t unflagged_compression_header_info; 
  memcpy((void*)&unflagged_compression_header_info, &send_package.header, sizeof(SGTP::header_data_t));

  //fill in last missing field automatically
  unflagged_compression_header_info.total_payload = unflagged_compression_header_info.texture_payload_size + unflagged_compression_header_info.geometry_payload_size;

  m_message_to_send_size_back = SGTP::HEADER_BYTE_SIZE + unflagged_compression_header_info.total_payload;
  memcpy((char*) (m_send_buffer_back) + SGTP::HEADER_BYTE_SIZE, send_package.message, unflagged_compression_header_info.total_payload);

  m_apply_jpeg_compression = m_apply_jpeg_compression_back;

  if(m_apply_jpeg_compression) {
    unflagged_compression_header_info.is_data_compressed = true;
  } else {
    unflagged_compression_header_info.is_data_compressed = false;    
  }

  unflagged_compression_header_info.package_reply_id = additional_header_info.package_reply_id;
  unflagged_compression_header_info.geometry_creation_time_in_ms = additional_header_info.geometry_creation_time_in_ms;
  unflagged_compression_header_info.timestamp = additional_header_info.timestamp;
  unflagged_compression_header_info.passed_microseconds_since_request = additional_header_info.passed_microseconds_since_request;


  memcpy(m_send_buffer_back, (void*)&unflagged_compression_header_info, SGTP::HEADER_BYTE_SIZE);
  m_is_new_message = true;
}



void GeometrySender::_send_loop(std::string const& server_socket) {
  // prepare zmq for SUBSCRIBER
  zmq::context_t ctx(1);                    // means single threaded
  zmq::socket_t send_socket(ctx, ZMQ_PUB);  // means a subscriber

  std::string const& tcp_socket_bind_string = "tcp://" + server_socket;
  
  send_socket.bind(tcp_socket_bind_string);

  std::cout << "opening server socket: " << tcp_socket_bind_string << std::endl;



  while (m_running) {

    // boost::this_thread::sleep_for(boost::chrono::milliseconds(2));
    //boost::mutex::scoped_lock lock(*m_mutex);
    if(true == m_is_new_message) {

      {
        boost::mutex::scoped_lock lock(*m_submission_mutex);
        std::swap(m_send_buffer, m_send_buffer_back);
        std::swap(m_message_to_send_size_back, m_message_to_send_size);
      }

      auto frame_time_start = std::chrono::system_clock::now();


      auto frame_time_stop = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed_compression_time = frame_time_stop - frame_time_start;
      m_compression_time_milliseconds_back = elapsed_compression_time.count();

      //std::string out_send_string = 
      //  "Sending @: " + std::to_string(m_message_to_send_size* 8 * 30 / (1024*1024.0))  + "Mbit/s\n";

      //std::cout << out_send_string;



      send_socket.send((void*)(m_send_buffer), m_message_to_send_size);
    }

    if(m_is_new_message) {
      m_is_new_message = false;
    }
  }

}
}
}
