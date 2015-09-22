# Text region identification

JNI wrapper to identify text boundaries in a PDF.

Dependencies:

- OpenCV
- Leptonica
- Ghostscript
- Tesseract

NOTE: Make sure to edit CMakeLists.txt to set your own ghostscript library location.

INSTALL:

> cmake .

> make

This creates the shared library object to load in the Java application.
Code currently expects a ProcessingRunnable class. I would suggest using a Runnable with a mutex on the function to create a viewable image file.
This is because ghostscript does not handle multiple threads very well (yet). It shouldn't affect performance since the main bottleneck is the Tesseract text extraction.

