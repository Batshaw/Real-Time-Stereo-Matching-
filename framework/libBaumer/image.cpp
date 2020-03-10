#include "image.h"
#include <iostream>


namespace GG {


image::image(std::string name):
  _data(0),
  _name(name),
  _width(0),
  _height(0),
  _quality(100)
{}


////////////////////////////////////////////////////////////////////////////////


  ///

image::image(std::string name, unsigned int width, unsigned int height):
  _data(0),
  _name(name),
  _width(width),
  _height(height),
  _quality(100)
{
  _data = new ggpixel [_width*_height];
  this->fill(0,0,0);
}


////////////////////////////////////////////////////////////////////////////////


  ///

/* virtual */
image::~image()
{
  if(_data)
  delete [] _data;
}


////////////////////////////////////////////////////////////////////////////////


  ///

/* virtual */
unsigned int
image::write()
{
  std::cerr << std::endl << "GG::image:          blind writing in Image::write() const";
  return 0;
}


////////////////////////////////////////////////////////////////////////////////


  ///

unsigned int
image::fill(unsigned char r, unsigned char g, unsigned char b)
{
  ggpixel* imagedata = this->getData();
  for (unsigned int y = 0 ; y < _height; ++y) {
    for (unsigned int x = 0 ; x < _width; ++x) {

      (*imagedata)(0) = r;
      (*imagedata)(1) = g;
      (*imagedata)(2) = b;

      ++imagedata;
    }
  }
  return 1;
}


////////////////////////////////////////////////////////////////////////////////


  ///

image*
image::select(int x1, int y1, int x2, int y2){
  unsigned int width = x2 - x1;
  unsigned int height = y2 -y1;
  image* s = new image("selection",width,height);
  ggpixel* sp = s->getData();
  ggpixel* mep = this->getData();
  mep += _width*y1 + x1;
  for(unsigned int y = 0; y < height; ++y){
    for(unsigned int x = 0; x < width; ++x){
      (*sp)(0) = (*mep)(0);
      (*sp)(1) = (*mep)(1);
      (*sp)(2) = (*mep)(2);
      ++sp;
      ++mep;
    }
    mep += _width - width;
  }
  return s;
}


////////////////////////////////////////////////////////////////////////////////


  ///

unsigned int
image::multiply(image* in)
{
  if(in->getWidth() != _width || in->getHeight() != _height)
    return 0;

  ggpixel* otherdata = in->getData();
  ggpixel* imagedata = this->getData();
  for (unsigned int y = 0 ; y < _height; ++y) {
    for (unsigned int x = 0 ; x < _width; ++x) {
      unsigned int tmp;

      tmp = (*imagedata)(0)*(*otherdata)(0);
      (*imagedata)(0) = tmp/255;

      tmp = (*imagedata)(1)*(*otherdata)(1);
      (*imagedata)(1) = tmp/255;

      tmp = (*imagedata)(2)*(*otherdata)(2);
      (*imagedata)(2) = tmp/255;

      ++imagedata;
      ++otherdata;
    }
  }
  return 1;  
}


////////////////////////////////////////////////////////////////////////////////


  ///

unsigned int
image::place(image *in, unsigned int col, unsigned row, unsigned char r, unsigned char g, unsigned char b)
{
  if(in->getWidth() + col > _width || in->getHeight() + row > _height)
    return 0;
  ggpixel* otherdata = in->getData();
  ggpixel* imagedata = this->getData();
  imagedata += row*_width + col;
  for(unsigned int y = 0; y < in->getHeight(); ++y){
    for(unsigned int x = 0; x < in->getWidth(); ++x){
      if(!((*otherdata)(0) == r && (*otherdata)(1) == g && (*otherdata)(2) == b)){
	(*imagedata)(0) = (*otherdata)(0);
	(*imagedata)(1) = (*otherdata)(1);
	(*imagedata)(2) = (*otherdata)(2);
      }
      ++imagedata;
      ++otherdata;
    }
    imagedata += _width - in->getWidth();
  }
  return 1;
}


////////////////////////////////////////////////////////////////////////////////


  ///

unsigned int
image::place(image *in, unsigned int col, unsigned row)
{
  if(in->getWidth() + col > _width || in->getHeight() + row > _height)
    return 0;
  ggpixel* otherdata = in->getData();
  ggpixel* imagedata = this->getData();
  imagedata += row*_width + col;
  for(unsigned int y = 0; y < in->getHeight(); ++y){
    for(unsigned int x = 0; x < in->getWidth(); ++x){
      (*imagedata)(0) = (*otherdata)(0);
      (*imagedata)(1) = (*otherdata)(1);
      (*imagedata)(2) = (*otherdata)(2);
      ++imagedata;
      ++otherdata;
    }
    imagedata += _width - in->getWidth();
  }
  return 1;
}


////////////////////////////////////////////////////////////////////////////////


  ///

unsigned int
image::place(image *in, unsigned int col, unsigned row, unsigned int percent)
{
  if(in->getWidth() + col > _width || in->getHeight() + row > _height)
    return 0;
  ggpixel* otherdata = in->getData();
  ggpixel* imagedata = this->getData();
  imagedata += row*_width + col;
  for(unsigned int y = 0; y < in->getHeight(); ++y){
    for(unsigned int x = 0; x < in->getWidth(); ++x){
      unsigned int tmp;
      
      tmp = ((100 - percent)*((unsigned int)(*imagedata)(0)) + percent*((unsigned int)(*otherdata)(0)))/100;
      (*imagedata)(0) = (unsigned char) tmp;

      tmp = ((100 - percent)*((unsigned int)(*imagedata)(1)) + percent*((unsigned int)(*otherdata)(1)))/100;
      (*imagedata)(1) = (unsigned char) tmp;

      tmp = ((100 - percent)*((unsigned int)(*imagedata)(2)) + percent*((unsigned int)(*otherdata)(2)))/100;
      (*imagedata)(2) = (unsigned char) tmp;

      ++imagedata;
      ++otherdata;
    }
    imagedata += _width - in->getWidth();
  }
  return 1;
}


////////////////////////////////////////////////////////////////////////////////


  ///

void
image::setQuality(unsigned int quality){
  _quality = quality;
}


////////////////////////////////////////////////////////////////////////////////


  ///

unsigned int
image::getQuality(){
  return _quality;
}


////////////////////////////////////////////////////////////////////////////////


  ///

std::string
image::getName() const
{
  return _name;
}


////////////////////////////////////////////////////////////////////////////////


  ///

void
image::setName(std::string name)
{
  _name = name;
}


////////////////////////////////////////////////////////////////////////////////


  ///

unsigned int
image::getWidth() const
{
  return _width;
}


////////////////////////////////////////////////////////////////////////////////


  ///

unsigned int
image::getHeight() const
{
  return _height;
}


////////////////////////////////////////////////////////////////////////////////


  ///

unsigned int
image::malloc(unsigned int width, unsigned int height)
{
  _width = width;
  _height = height;
  if(_data)
    delete _data;
  _data = new ggpixel [_width*_height];
  return 1;
}


////////////////////////////////////////////////////////////////////////////////


  ///

const ggpixel&
image::operator() (unsigned int col,unsigned int row) const
{
  return _data[_width*row+col];
}


////////////////////////////////////////////////////////////////////////////////


  ///

ggpixel&
image::operator() (unsigned int col,unsigned int row)
{
  return _data[_width*row+col];
}


////////////////////////////////////////////////////////////////////////////////


  ///

ggpixel*
image::getData()
{
  return _data;
}

} // namespace GG
