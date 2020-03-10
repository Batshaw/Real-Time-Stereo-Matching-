#include "jpg.h"



namespace GG {
          

////////////////////////////////////////////////////////////////////////////////


  ///

jpg::jpg(unsigned int width, unsigned int height):
  image("Empty.jpg")
{
  this->malloc(width,height);
}


////////////////////////////////////////////////////////////////////////////////


  ///

jpg::jpg(std::string filename):
  image(filename)
{ 
  imageRGB img = read_JPEG_file (filename.c_str());
  this->malloc(img.width,img.height);
  ggpixel* imagedata = this->getData();
  for (int y = 0 ; y < img.height; ++y) {
    for (int x = 0 ; x < img.width; ++x) {
      unsigned char* ggpixel = &img.data[3*(x + img.width*y)];
      (*imagedata)(0) = ggpixel[0];
      (*imagedata)(1) = ggpixel[1];
      (*imagedata)(2) = ggpixel[2];
      ++imagedata;
    }
  }
}


////////////////////////////////////////////////////////////////////////////////


  ///

/* virtual */
jpg::~jpg()
{}


////////////////////////////////////////////////////////////////////////////////


  ///

/* virtual */
unsigned int
jpg::write()
{
  imageRGB img;
  img.width = this->getWidth();
  img.height = this->getHeight();
  img.data = new unsigned char [3*img.width*img.height];
  ggpixel* imagedata = this->getData();
  for (int y = 0 ; y < img.height; ++y) {
    for (int x = 0 ; x < img.width; ++x) {
      unsigned char* ggpixel = &img.data[3*(x + img.width*y)];
      ggpixel[0] = (*imagedata)(0);
      ggpixel[1] = (*imagedata)(1);
      ggpixel[2] = (*imagedata)(2);
      ++imagedata;
    }
  }
  write_JPEG_file (this->getName().c_str(), this->getQuality(), img);
  delete [] img.data;
  return 1;
}




////////////////////////////////////////////////////////////////////////////////


  ///




////////////////////////////////////////////////////////////////////////////////


  ///




////////////////////////////////////////////////////////////////////////////////


  ///




} // namespace GG
