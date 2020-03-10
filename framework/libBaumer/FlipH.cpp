#include <FlipH.h>
#include <iostream>

namespace GG {

FlipH::FlipH(image* im)
{
  unsigned int width  = im->getWidth();
  unsigned int height = im->getHeight();
  for (int y = 0 ; y < height; ++y) {
    for (int x = 0 ; x < width/2; ++x) {
      
      ggpixel tmp        = (*im)(x,y);
      
      (*im)(x,y)       = (*im)(width-x,y);
      (*im)(width-x,y) = tmp;
    }
  }
}



FlipH::~FlipH()
{}

}