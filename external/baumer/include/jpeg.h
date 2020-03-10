#define _WINDOWS_H

#ifndef JPEG_H
#define JPEG_H
/* Simple JPEG load/save interface,
 * based on examplecode in IJG's JPEG-library
 *
 * Author Fredrik Orderud, 2005
 */

#ifndef UCHAR
#define UCHAR
typedef unsigned char uchar;
#endif

// struct for handling images
typedef struct {
	uchar* data; //pixel data in RGB format. sizeof(data) == 3 * width * height;
	int width, height;
} imageRGB;


/******************** JPEG (DE)COMPRESSION INTERFACE *******************/

/* Compress image into JPEG, and save it to disk
   quality must be in range 0-100 */
void write_JPEG_file (const char * filename, int quality, imageRGB img);

/* Load and decompress JPEG image from disk */
imageRGB read_JPEG_file (const char * filename);

#endif
