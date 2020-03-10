#include <FlipV.h>
#include <iostream>

namespace GG {

FlipV::FlipV(image* im)
{
  unsigned int width  = im->getWidth();
  unsigned int height = im->getHeight();
  for (int y = 0 ; y < height/2; ++y) {
    for (int x = 0 ; x < width; ++x) {
      
      ggpixel tmp         = (*im)(x,y);
      (*im)(x,y)        = (*im)(x,height-y-1);
      (*im)(x,height-y-1) = tmp;
      
    }
  }
}



FlipV::~FlipV()
{}

}