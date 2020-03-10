#include "BSystem.h"
#include <BCamera.h>


#include <bgapi.hpp>



#include <cstdlib>
#include <iostream>

namespace baumer {


  /*static*/ int BSystem::s_system_count = -1;
  /*static*/ int BSystem::s_created_systems = 0;

  BSystem::BSystem()
    :
    m_system(0),
    m_cameras()
  {


  }


  BSystem::~BSystem(){

    for(unsigned int i = 0; i < m_cameras.size(); ++i){
      if(0 != m_cameras[i])
	m_cameras[i]->stop();
    }



    m_system->release();

    --s_created_systems;
    if(0 == s_created_systems){
      std::cout << "libBaumer::BSystem::~BSystem() last system deleted...dont know what to do yet..." << std::endl;
    }

  }



  bool
  BSystem::init(){


    int res = false;
    if(0 > s_system_count){ // we have to count the available systems
      res = BGAPI::countSystems( &s_system_count);
      std::cout << "libBaumer::BSystem::init() found " << s_system_count << " systems" << std::endl;
    }

    // dont now if that is correct yet
    /* original it was
      int sys = 0;
      int res = BGAPI::createSystem( sys, &m_system);
    */
    BGAPI::createSystem( s_created_systems, &m_system);
    std::cout << "libBaumer::BSystem::init() created system " << s_created_systems << std::endl;

    m_system->open();
    std::cout << "libBaumer::BSystem::init() opend system " << s_created_systems << std::endl;

    ++s_created_systems;



    //COUNTING AVAILABLE CAMERAS
    int camera_count = 0;
    m_system->countCameras(&camera_count);
    std::cout << "libBaumer::BSystem::init() found " << camera_count << " cameras" << std::endl;


    for(int i = 0; i < camera_count; ++i){
      m_cameras.push_back(0);
    }

    return true;
  }



  unsigned int
  BSystem::getNumCameras() const{
    return m_cameras.size();
  }


  void
  BSystem::openAll(){
    for(unsigned int i = 0; i < getNumCameras(); ++i)
        getCamera(i);
  }



  BCamera*
  BSystem::getCamera(unsigned int num, bool rgb, bool fastest){
    if ((num + 1) > m_cameras.size()){
      std::cerr << "ERROR: libBaumer::BSystem::getCamera() camera " << num << " not available" << std::endl;
      return 0;
    }

    if(0 == m_cameras[num]){
      int res = 0;
      BGAPI::Camera * pCamera = NULL;

      m_system->createCamera( (int) num, &pCamera);
      std::cout << "libBaumer::BSystem::getCamera() created camera: " << num << std::endl;

      m_cameras[num] = new BCamera(pCamera);

      
      pCamera->open();



      std::cout << "libBaumer::BSystem::getCamera() opened camera: " << num << std::endl;
      if(rgb){
	// 1 /*Full Frame @ 30 Hz*/
	// 0 /*Full Frame @ 20 Hz HQ*/
	pCamera->setImageFormat(1);
	m_cameras[num]->setNumChannels(3);
	pCamera->setPixelFormat(BGAPI_PIXTYPE_BAYRG8);
	std::cout << "libBaumer::BSystem::getCamera() created camera as rgb" << std::endl;
      }
      else{

	if(fastest)
	  pCamera->setImageFormat(3); // Binning 2x2
	else
	  pCamera->setImageFormat(1); // no binning
	
	m_cameras[num]->setNumChannels(1);
	pCamera->setPixelFormat(BGAPI_PIXTYPE_MONO8);
	std::cout << "libBaumer::BSystem::getCamera() created camera as mono" << std::endl;
      }


      //QUEUED MODE / USING INTERNAL BUFFER
      const int numBuffers = 2;
      pCamera->setDataAccessMode( BGAPI_DATAACCESSMODE_QUEUEDINTERN, numBuffers );



      BGAPI::Image * pImage = NULL;
      for(int i = 0; i < numBuffers; ++i){

	m_system->createImage( &pImage, false );
	pCamera->setImage( pImage );



	//SETTING A DESTINATION FORMAT
	if(rgb){
	  pImage->setDestinationPixelFormat(BGAPI_PIXTYPE_RGB8_PACKED);
	}
	else{
	  pImage->setDestinationPixelFormat(BGAPI_PIXTYPE_MONO8);
	}

#if 0
	res =  pCamera->setTriggerSource( BGAPI_TRIGGERSOURCE_HARDWARE1 );
      if( res != BGAPI_RESULT_OK )
        {
            printf("BGAPI::Camera::setTriggerSource Errorcode: %d\n", res);
        }
#endif


      }
      m_cameras[num]->start();


    }
    return m_cameras[num];
  }

  BCamera*
  BSystem::getCamera(const char* serial){
    for(unsigned int i = 0; i < getNumCameras(); ++i){
      BCamera* c = getCamera(i);
      if(c->getSerial() == std::string(serial))
           return c;
    }
    return 0;
  }



}
