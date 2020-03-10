#ifndef PNM_H
#define PNM_H

#include <image.h>

namespace GG {

class PNM : public image{

 public:

  PNM(unsigned int width, unsigned int height);
  PNM(std::string filename);
  /* virtual */ ~PNM();

  /* virtual */ unsigned int write();
};

}
#endif // #ifndef PNM_H
