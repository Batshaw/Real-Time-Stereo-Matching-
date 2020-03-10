#ifndef BAUMER_SYSTEM_H
#define BAUMER_SYSTEM_H


#include <vector>

namespace BGAPI {
  class System;
}

namespace baumer{
  class BCamera;
  class BSystem{



  public:
    BSystem();
    ~BSystem();
    bool init();

    unsigned int getNumCameras() const;
    void openAll();
    BCamera* getCamera(unsigned int num = 0, bool rgb = true, bool fastest = true);
    BCamera* getCamera(const char* serial);

  private:

    BGAPI::System* m_system;
    std::vector <BCamera* > m_cameras;
    

    static int s_system_count;
    static int s_created_systems;

  };


}


#endif // #ifndef BAUMER_SYSTEM_H
