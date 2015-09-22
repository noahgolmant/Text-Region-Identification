cmake_minimum_required(VERSION 3.0.2)
project(text_region_identification)

#set(CMAKE_BUILD_TYPE Release)

find_package( PNG REQUIRED )
find_package( OpenCV REQUIRED )

set(GS_LOCATION /your/ghostscript/library)
set(JNI_LOCATION /usr/lib/jvm/jdk1.8.0_45)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(${JNI_LOCATION}/include ${JNI_LOCATION}/include/linux)
include_directories(/usr/include/libpng12)
include_directories(/usr/local/include)

link_directories(${GS_LOCATION}/sobin)
link_directories(/usr/local/lib/tesseract)
link_directories(/usr/local/lib/leptonica)

set(SOURCE_FILES GSDocumentConversion.cpp ProcessingRunnable.cpp ProcessingRunnable.h)
add_library(text_region_identification SHARED ${SOURCE_FILES} ImageProcessing.cpp)

target_link_libraries(text_region_identification png12 ${OpenCV_LIBS} gs tesseract lept)
