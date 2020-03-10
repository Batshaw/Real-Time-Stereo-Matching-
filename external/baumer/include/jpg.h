#ifndef GG_JPG_H
#define GG_JPG_H

#include "image.h"
#include "jpeg.h"


namespace GG {


  /// jpeg image

class jpg : public image{
 public:
  jpg(unsigned int width, unsigned int height);
  jpg(std::string filename);
  /* virtual */ ~jpg();
  /* virtual */ unsigned int write();
};

} // namespace GG

#endif // #ifndef GG_JPG_H
