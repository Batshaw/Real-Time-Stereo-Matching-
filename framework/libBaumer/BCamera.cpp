#include "BCamera.h"


#include <clock.h>


#include <map>
#include <iostream>
#include <cmath>

#include <sstream>
#include <iomanip>

namespace baumer {

  template <class T>
  inline std::string
  toString(T value)
  {
    std::ostringstream stream;
    stream << std::setfill ('0') << std::setw (4) << value;
    return stream.str();
  }

  static std::map<BGAPI::Camera*, BCamera*> s_camera_mapping;

  static BGAPI_RESULT BGAPI_CALLBACK imageCallback( void * callBackOwner, BGAPI::Image * pImage ){
    BGAPI::Camera * pCamera = (BGAPI::Camera*)callBackOwner;
    return s_camera_mapping[pCamera]->handleImage(pCamera,pImage);
  }



  BCamera::BCamera(BGAPI::Camera* pCamera)
    : 
    m_serial(),
      m_gain(1.0), 
      m_num_channels(3),
      m_camera(pCamera),
      m_format(),
      m_software_counter(0),
      m_hardware_counter(0),
      m_transform_buffer_length(0),
      m_front(0),
      m_back(0),
      m_mutex_lock(),
      m_ts(),
      m_smoother(20),

      m_capturetime(),
      m_avg_fps()
  {}

  BCamera::~BCamera()
  {}


  bool
  BCamera::start(){
    int res = 0;
    s_camera_mapping[m_camera] = this;
    m_camera->registerNotifyCallback( m_camera, (BGAPI::BGAPI_NOTIFY_CALLBACK) &imageCallback );
    m_camera->setStart( true );


    {
      

      BGAPI_FeatureState state;
      state.cbSize = sizeof( BGAPI_FeatureState );
      BGAPIX_CameraImageFormat cformat;
      cformat.cbSize = sizeof ( BGAPIX_CameraImageFormat );
      BGAPIX_TypeListINT formatlist;
      formatlist.cbSize = sizeof( BGAPIX_TypeListINT );
      res = m_camera->getImageFormat( &state, &formatlist );
      if( res != BGAPI_RESULT_OK ){
	std::cerr << "BGAPI::Camera::getImageFormat Errorcode" << res << std::endl;
	return false;
      }
      res = m_camera->getImageFormatDescription( formatlist.current, &cformat );

      m_format.iSizeX = cformat.iSizeX;
      m_format.iSizeY = cformat.iSizeY;

      BGAPIX_CameraInfo camdeviceinfo;
      camdeviceinfo.cbSize = sizeof( BGAPIX_CameraInfo );
      res = m_camera->getDeviceInformation( &state, &camdeviceinfo );
      if( res != BGAPI_RESULT_OK ){
          std::cerr << "BGAPI::Camera::getDeviceInformation Errorcode" << res << std::endl;
         return false;
      }

      m_serial = camdeviceinfo.serialNumber;


      std::cerr << "width: " << getWidth() << " height: " << getHeight() << std::endl;


    }




    return true;
  }

  bool
  BCamera::stop(){
    m_camera->setStart( false );
    delete [] m_front;
    m_front = 0;
    delete [] m_back;
    m_back = 0;
    return true;
  }

  BGAPI_RESULT
  BCamera::handleImage(BGAPI::Camera * pCamera,BGAPI::Image *pImage ){


    BGAPI_RESULT res = BGAPI_RESULT_OK;


    pImage->getNumber( &m_software_counter, &m_hardware_counter );



    
#if 0
    std::cerr << this << ": " << m_software_counter << std::endl;
    unsigned char* imagebuffer = NULL;

    pImage->get( &imagebuffer );
    //Now you have the Imagebuffer and can do with it whatever you want

#endif


    if(0 == m_transform_buffer_length){
      pImage->getTransformBufferLength(&m_transform_buffer_length);
      m_front = new unsigned char[m_transform_buffer_length + 3 * 4 * m_format.iSizeX];
      m_back = new unsigned char[m_transform_buffer_length  + 3 * 4 * m_format.iSizeX];
      m_ts = sensor::clock::time();
    }
    res = pImage->doTransform(m_back, m_transform_buffer_length);
    if( res != BGAPI_RESULT_OK ){
      std::cerr << "ERROR in BCamera::handleImage() doTransform Error code: " << res << std::endl;
    }

    //after you are ready with this image, return it to the camera for the next image
    res = pCamera->setImage( pImage );
    if( res != BGAPI_RESULT_OK )
      std::cerr << "ERROR in BCamera::handleImage() setImage failed, error: " << res << std::endl;

    { // swap front back
      boost::mutex::scoped_lock l(m_mutex_lock);
      unsigned char* tmp = m_front;
      m_front = m_back;
      m_back = tmp;

      sensor::timevalue ts_now(sensor::clock::time());
      m_capturetime = (ts_now - m_ts).msec();
      m_avg_fps = std::floor(1000.0/ m_smoother((int) (m_capturetime)));
      m_ts = ts_now;
    }






    return res;

  }


  unsigned int
  BCamera::getWidth() const{
    return m_format.iSizeX;
  }

  unsigned int
  BCamera::getHeight() const{
    return m_format.iSizeY + 4;
  }


  boost::mutex&
  BCamera::getMutexLock(){
    return m_mutex_lock;
  }

  unsigned char*
  BCamera::capture(){


    return m_front;
  }

  unsigned int
  BCamera::getCaptureTime() const{
    return m_capturetime;
  }

  float
  BCamera::getAvgFps() const{
    return m_avg_fps;
  }

  void
  BCamera::setNumChannels(unsigned c){
    m_num_channels = c;
  }

  unsigned
  BCamera::getNumChannels() const{
    return m_num_channels;
  }

  void
  BCamera::setGain(float g){
    m_gain = g;

    m_camera->setGain(m_gain);
    //std::cerr << this << " m_gain: " << m_gain << std::endl;
#if 0 
    m_camera->setGainAutoMode( BGAPI_AUTOMATICMODE_CONTINUOUS );
    BGAPIX_TypeRangeFLOAT brightinpercent;
    brightinpercent.cbSize = sizeof ( BGAPIX_TypeRangeFLOAT );
    m_camera->setExposureAutoMode( BGAPI_AUTOMATICMODE_CONTINUOUS );

    float targetbrightness = 100;
    m_camera->setExposureAutoBrightness( targetbrightness );
#endif
    //m_camera->setWhiteBalance( false, 0,0,0 );

    BGAPI_FeatureState state;
    BGAPIX_TypeRangeFLOAT red;
    BGAPIX_TypeRangeFLOAT green;
    BGAPIX_TypeRangeFLOAT blue;
    BGAPIX_TypeROI roi;
    m_camera->getWhiteBalance( &state, &red, &green, &blue, &roi);
    //std::cerr << (int) state.bIsEnabled << " : " << red.current << " " << green.current << " " << blue.current << std::endl;
  }

  void
  BCamera::setExposureTime(float in_exposure_time) {
    std::cout << "Dummy Exposure time function called " << in_exposure_time << std::endl;
    m_camera->setExposure(in_exposure_time);
  }


  float
  BCamera::getGain() const{
    return m_gain;
  }


  const char*
  BCamera::getSerial() const{
    return m_serial.c_str();
  }


}
