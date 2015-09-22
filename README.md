# Text region identification

Identifies text boundaries in a PDF.

Dependencies:
- OpenCV
- Leptonica
- Ghostscript


NOTE: Make sure to edit CMakeLists.txt to set your own ghostscript library location.

INSTALL:

> cmake .
> make

USAGE:

> ./text_region_identification <filename>.pdf
