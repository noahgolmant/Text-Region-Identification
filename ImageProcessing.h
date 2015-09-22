
#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H
#ifdef __cplusplus
extern "C" {
#endif

//#ifndef DEBUG
//  #define DEBUG
//#endif

#define GS_SUCCESS 1
#define GS_FAILURE -1

extern int pageCount;
extern int convertPDF(const char* in, const char* out);

extern int createViewableImageFile(char*);
extern int processImageFile(char*); 
#ifdef __cplusplus
}
#endif
#endif
