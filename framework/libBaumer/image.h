#ifndef IMAGE_H
#define IMAGE_H

#include <ggpixel.h>
#include <string>


namespace GG {


  /// image base class

class image
{
  public:
  
    image(std::string name);
    image(std::string name, unsigned int width, unsigned int height);
  
  
    virtual ~image();
  
    ggpixel* getData();
  
    std::string getName() const;
    void        setName(std::string name);
  
    void         setQuality(unsigned int quality);
    unsigned int getQuality();
  
    unsigned int getWidth() const;
    unsigned int getHeight() const;
    
  
  
    virtual unsigned int write();
    image*               select(int x1, int y1, int x2, int y2);
    
    unsigned int fill(unsigned char r, unsigned char g, unsigned char b);
    unsigned int place(image *in, unsigned int col, unsigned int row, unsigned char r, unsigned char g, unsigned char b);
    unsigned int place(image *in, unsigned int col, unsigned int row);
    unsigned int place(image *in, unsigned int col, unsigned int row, unsigned int percent);
    unsigned int multiply(image* in);
    const ggpixel& operator() (unsigned int col,unsigned int row) const;
    ggpixel& operator() (unsigned int col,unsigned int row);
  


  protected:
    unsigned int malloc(unsigned int width, unsigned int height);

  private:
    ggpixel* _data;
    std::string _name;
    unsigned int _width;
    unsigned int _height;
    unsigned int _quality;

};

} // namespace GG

#endif // #ifndef IMAGE_H
