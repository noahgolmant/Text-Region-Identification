cmake_minimum_required(VERSION 3.0.2)
project(text_region_identification)

#set(CMAKE_BUILD_TYPE Release)

find_package( PNG REQUIRED )
find_package( OpenCV REQUIRED )

set(GS_LOCATION /home/noah/Dev/c-libs/ghostscript-9.16)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(/usr/include/libpng12)
include_directories(/usr/local/include)

link_directories(${GS_LOCATION}/sobin)
link_directories(/usr/local/lib/leptonica)

set(SOURCE_FILES GSDocumentConversion.cpp ImageProcessing.cpp ImageProcessing.h)
add_executable(text_region_identification ${SOURCE_FILES})

target_link_libraries(text_region_identification png12 ${OpenCV_LIBS} gs lept)
