#include <PNM.h>
#include <iostream>
#include <fstream>

namespace GG {

PNM::PNM(unsigned int width, unsigned int height):
  image("Empty")
{
  this->malloc(width,height);
}


// evtl. kommentar aus pnm file rausgrillen
PNM::PNM(std::string filename):
  image(filename)
{
  std::ifstream is(filename.c_str());
  std::string magic;
  unsigned int width, height, maxcolor;
  unsigned char commenttest;
  
  is >> magic;
  if (magic == "P6") {
  
    is >> width >> height >> maxcolor;
    // zeilenumbruch...
    is.get();

    this->malloc(width,height);
    ggpixel* imagedata = this->getData();
    for (int y = 0 ; y < height; ++y) {
      for (int x = 0 ; x < width; ++x) {
        (*imagedata)(0) = is.get();
	(*imagedata)(1) = is.get();
	(*imagedata)(2) = is.get();
	++imagedata;
      }
    }
  }
}

/* virtual */
PNM::~PNM()
{}


// evtl. kommentar reinschreiben
/* virtual */ unsigned int
PNM::write()
{
  if(this->getName() == "")
    return 0;

  std::ofstream of(this->getName().c_str());
  std::string magic = "P6";
  unsigned int width = this->getWidth();
  unsigned int height = this->getHeight();
  unsigned int maxcolor = 255;
  of << magic << std::endl;
  of << width << " " << height << std::endl;

  of << maxcolor << std::endl;

  ggpixel* imagedata = this->getData();
  for (int y = 0 ; y < height; ++y) {
    for (int x = 0 ; x < width; ++x) {
      of << (*imagedata)(0);
      of << (*imagedata)(1);
      of << (*imagedata)(2);
      ++imagedata;
    }
  }
  return 1;
}

}

