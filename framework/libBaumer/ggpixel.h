#ifndef ggpixel_H
#define ggpixel_H



namespace GG {


  /// RGB ggpixel


class ggpixel
{

  public:
    ggpixel();
    ggpixel(unsigned int r, unsigned int g, unsigned int b);
    ggpixel( const ggpixel& );
    ~ggpixel();

    const unsigned char& operator () (unsigned char rgb) const;
    unsigned char& operator () (unsigned char rgb);
    const ggpixel& operator = (const ggpixel& rhs);
    unsigned char* getData();
    
      
    /// indexing
    unsigned char& operator[](unsigned int);
    const unsigned char& operator[](unsigned int) const;


 private:
  unsigned char _data[3];
};


}; // namespace GG


#endif // ggpixel_H
