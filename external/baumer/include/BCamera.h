#ifndef BAUMER_CAMERA_H
#define BAUMER_CAMERA_H

#include <bgapi.hpp>

#include <Smoother.h>
#include <timevalue.h>


#include <boost/thread/mutex.hpp>
#include <string>

namespace BGAPI {
  class Camera;
}

namespace baumer {
  class BSystem;
  class BCamera{

    friend class BSystem;

  public:
    class ImageFormat{
    public:
      int id;
      std::string name;
      unsigned int iOffsetX;
      unsigned int iOffsetY;
      unsigned int iScaleRoiX;
      unsigned int iScaleRoiY;
      unsigned int iSizeX;
      unsigned int iSizeY;
    };

  private:
    BCamera(BGAPI::Camera* pCamera);
    ~BCamera();

    bool start();

    bool stop();

  public:
    BGAPI_RESULT handleImage(BGAPI::Camera * pCamera, BGAPI::Image *pImage);

    void setSaveFrames(bool truefalse);

    unsigned int getWidth() const;
    unsigned int getHeight() const;

    boost::mutex& getMutexLock();
    unsigned char* capture();

    unsigned int getCaptureTime() const;
    float getAvgFps() const;

    void setNumChannels(unsigned c);
    unsigned getNumChannels() const;

    void setGain(float g);
    float getGain() const;


    const char* getSerial() const;

  private:
    std::string m_serial;
    float m_gain;
    unsigned m_num_channels;
    BGAPI::Camera* m_camera;
    ImageFormat    m_format;
    int   m_software_counter;
    int   m_hardware_counter;
    
    int m_transform_buffer_length;
    unsigned char* m_front;
    unsigned char* m_back;
    boost::mutex m_mutex_lock;

    sensor::timevalue m_ts;
    sensor::Smoother  m_smoother;

    unsigned int m_capturetime;
    float m_avg_fps;
    bool m_save_frames;
  public:
    void saveFrame(const char* filename = NULL);
  };

}





#endif // #ifndef  BAUMER_CAMERA_H
