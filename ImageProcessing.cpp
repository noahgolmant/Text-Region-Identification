//
// Created by noah on 7/15/15.
//
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>
#include "ImageProcessing.h"

/* UTILITY FUNCTIONS */
static inline int max(int, int);
static inline bool fileExists(char *);

/**
 * OPENCV FUNCTIONS
 */
int createViewableImageFile(char*);
static int createImageWithOverlay(cv::Mat, char*);
int processImageFile(char*); 

/**
 * IMAGE PROCESSING DATA
 */

/*------------- SOBEL OPERATOR CONSTANTS ---------------------*/
/* bit depth of the sobel operator on our image */
static const int   SOBEL_DEPTH       = CV_8U;
/* the sobel operator is only concerned with delta x */
static const int   SOBEL_DX          = 1;
static const int   SOBEL_DY          = 0;
/* 3x3 area kernel */
static const int   SOBEL_KERNEL_SIZE = 9;
/* we do not change the actual size/dimensions of the image */
static const float SOBEL_SCALE       = 1.0;
/* do not adjust the actual intensity of the pixels */
static const float SOBEL_DELTA       = 0.0;
/*------------------------------------------------------------*/

/*---------------- MORPHOLOGICAL ELEMENTS --------------------*/
static const int MORPH_ELEM        = 2; /* oval */
static const int MORPH_KERNEL_SIZE = 9;
/*------------------------------------------------------------*/

/*---------- RECTANGLE IDENTIFICATION CONSTRAINTS ------------*/
static const int MIN_RECT_AREA = 50;
/*------------------------------------------------------------*/

/*----------- POLY APPROXIMATION CONSTANTS -------------------*/
/* max distance of poly approximation from the real value */
static const int  APPROX_EPSILON        = 3;
/* the beginning and ends of the poly curve are connected,
 * meaning we form an oval that can be converted rectangle based
 * on its boundaries */
static const bool APPROX_CLOSED         = false;
/* expand our rectangle to capture all of the text */
static const double  APPROX_COMPENSATION_X = 1.05;
static const double  APPROX_COMPENSATION_Y = 1.1;
/*------------------------------------------------------------*/

/* PRECISION CONSTANTS */
static const int NUM_COORD_DIGITS = 7;

/* MINIMUM TEXT SIZES */



/**
  * Computes the maximum value between two integers
  */
static inline int max(int a, int b) {
    if(a >= b)
        return a;
    else
        return b;
}

/**
  * Determines whether or not a file of filename str exists
  */
static inline bool fileExists(char *str) {
    struct stat buffer;
    return (stat (str, &buffer) == 0);
}


static inline long getTimeMilliseconds() {
    timeval tv;
    gettimeofday(&tv, NULL);

    return tv.tv_usec / 1000;
}

/**
 * function to retrieve access to the viewable image file that contains
 * the concatenation of the pages of the original document.
 *
 * If the file already exists, it returns that. Otherwise, it creates a new one
 * with a String input of the absolute path of the document.
 *
 * @param filestr String representation of the absolute path of the document
 */
int createViewableImageFile(char* filestr, char* out) {

    // call the ghostscript conversion process to
    // create separate images per page of the document
    int convertRet = convertPDF(filestr, "temp.bmp");
    if(convertRet != GS_SUCCESS) {
        fprintf(stderr, "Failed to convert PDF.\n");
        return 1;
    }

    // check to make sure there are actual pages to process
    if(pageCount == 0 /* default value of pageCount */) {
        fprintf(stderr, "No pages to process.\n");
        return 1;
    }

    /* Concatenate the page images into a single image */

    // Create the final concatenated image Mat and the Mat array to hold each page.
    cv::Mat concatenatedImg;
    cv::Mat* pageMats = new cv::Mat[pageCount];

    // Access the temporarily stored page image through its file name.
    // It is formatted over every iteration to get the correct file per page number.
    const char* imgStr = "%dtemp.bmp";
    char formattedImgStr[512];

    // Initialize total width for later initialization of the concatenated Mat
    int concatenatedImgWidth  = 0;

    // Begin page concatenation
    int i;
    // Note that ghostscript pagecount is not 0-based, so we start at 1
    for(i = 1; i <= pageCount; i++) {
        // Get the correct filename for page i
        sprintf(formattedImgStr, imgStr, i);

        // make sure that the file exists
        if(!fileExists(formattedImgStr)) {
            fprintf(stderr, "%s was not created successfully\n", formattedImgStr);
            return 1;
        }

        // store the page's image in the page image array (loading it into memory)
        pageMats[i - 1] = cv::imread(formattedImgStr, CV_LOAD_IMAGE_UNCHANGED);
        unlink(formattedImgStr);
        //cv::bitwise_not(pageMats[i-1], pageMats[i-1]);
        // check if the OpenCV failed to read the image for some reason
        if(pageMats[i-1].empty()) {
            fprintf(stderr, "Failed to create cv::Mat for %s\n", formattedImgStr);
            fflush(stderr);
            return 1;
        }

        // Update the concatenated width for its later initialization
        concatenatedImgWidth = max(concatenatedImgWidth, pageMats[i-1].cols);
    }

    // Initialize and fill the concatenated image Mat
    concatenatedImg = cv::Mat(0, concatenatedImgWidth, CV_32F);
    for(i = 0; i < pageCount; i++) {
        concatenatedImg.push_back(pageMats[i]);
    }

    // Create the final viewable image file as a concatenation of the previous images
    strcpy(out, filestr);
    strcat(out, ".bmp");

    // Write the file from memory
    cv::imwrite(out, concatenatedImg);

    // return to java control
    return 0;
}

static int createImageWithOverlay(cv::Mat imageToProcess /* unedited image */,
                                            char* filestr) {
    cv::Mat processingMat(imageToProcess);
    cv::Mat displayableMat(imageToProcess);
    #ifdef DEBUG

    fprintf(stdout, "\nSOBEL_DEPTH: %d\tSOBEL_DX: %d\tSOBEL_DY: %d\n"
                    "SOBEL_KERNEL_SIZE: %d\tSOBEL_SCALE: %f\tSOBEL_DELTA: %f\n",
                SOBEL_DEPTH, SOBEL_DX, SOBEL_DY, SOBEL_KERNEL_SIZE, SOBEL_SCALE, SOBEL_DELTA);

    long startTime = getTimeMilliseconds();

    #endif

    /* SOBEL PROCESSING */
    /* calculates image derivative, i.e. the gradients relative to the kernel size */
    /* maximum/minimum identification is used for edge detection */
    cv::cvtColor(processingMat, processingMat, CV_RGB2GRAY);

    cv::Sobel(processingMat, processingMat,
              SOBEL_DEPTH,
              SOBEL_DX,
              SOBEL_DY,
              SOBEL_KERNEL_SIZE,
              SOBEL_SCALE,
              SOBEL_DELTA,
              cv::BORDER_DEFAULT);

    #ifdef DEBUG
    long time = getTimeMilliseconds() - startTime;
    fprintf(stdout, "--- SOBEL TIME: %ldms ---\n", abs(time));
    fflush(stdout);
    startTime = getTimeMilliseconds();
    #endif

    /* OTSU THRESHOLD */
    /* binarizes an image (light gray -> white, dark gray -> black)
     * based on relative pixel intensities in the kernel areas */
    //cv::cvtColor(processingMat, processingMat, CV_8UC1);

    cv::threshold(processingMat, processingMat, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

    #ifdef DEBUG
    time = getTimeMilliseconds() - startTime;
    fprintf(stdout, "--- OTSU TIME: %ldms ---\n", abs(time));
    fflush(stdout);
    startTime = getTimeMilliseconds();
    #endif

    /* MORPHOLOGICAL CLOSING */
    cv::Mat structuringElement = cv::getStructuringElement(
        MORPH_ELEM,
        cv::Size(2 * MORPH_KERNEL_SIZE + 1, 2 * MORPH_KERNEL_SIZE + 1),
        cv::Point(MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
    );
    // creates a rectangular element to identify in our image
    // creates filled rectangles where text might be in our image
    cv::morphologyEx(processingMat, processingMat, cv::MORPH_CLOSE, structuringElement);

    #ifdef DEBUG
    time = getTimeMilliseconds() - startTime;
    fprintf(stdout, "--- MORPHOLOGICAL OPERATIONS TIME: %ldms ---\n", abs(time));
    fflush(stdout);
    startTime = getTimeMilliseconds();
    #endif

    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i>              hierarchy;

    cv::findContours(processingMat, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

    #ifdef DEBUG
    time = getTimeMilliseconds() - startTime;
    fprintf(stdout, "--- FIND CONTOURS TIME: %ldms ---\n", abs(time));
    fflush(stdout);
    startTime = getTimeMilliseconds();
    #endif

    std::vector<cv::Rect> rectangles;
    for(int i = 0; i < contours.size(); i++) {
        cv::approxPolyDP(contours.at(i), contours.at(i), APPROX_EPSILON, APPROX_CLOSED);

        cv::Rect contRect = cv::boundingRect(contours.at(i));

        // expand width so we don't cut text off
        double newWidth = contRect.width * APPROX_COMPENSATION_X;
        double newHeight = contRect.height * APPROX_COMPENSATION_Y;

        contRect.x -= newWidth - contRect.width;
        contRect.y -= newHeight - contRect.height;

        contRect.width = newWidth;
        contRect.height = newHeight;

        rectangles.push_back(contRect);
        rectangles.push_back(contRect);

    }


    /* 1 = group rectangles that have at least one close enough in the cluster
    .*     where the cluster size is determined by epsilon (0.05)
     */
    cv::groupRectangles(rectangles, 1, 0.05);

    /* draw rectangles onto image */
    for(int i = 0; i < rectangles.size(); i++) {
        cv::rectangle(displayableMat, rectangles.at(i), cv::Scalar(0,0,0), 3);
    }
    
    // Create image with visible text boundaries
    char displayableImgFileStr[256];
    strcpy(displayableImgFileStr, filestr);
    strcat(displayableImgFileStr, ".overlay.bmp");

    // Write the file from memory
    cv::imwrite(displayableImgFileStr, displayableMat);
    fprintf(stdout, "Wrote image with text region overlay to: %s\n", displayableImgFileStr);

    #ifdef DEBUG
    fprintf(stdout, "--- GROUP RECTANGLES TIME: %ldms ---\n", abs(time));
    fflush(stdout);
    #endif

    return 0;
}

int processImageFile(char* filestr) {
    // check to see if the file actually exists, i.e. it's already been processed
    if(!fileExists(strdup(filestr))) {
        fprintf(stderr, "Tried to process an image file that doesn't exist: %s\n", filestr);
        return 1;
    }
    char* img_filestr = (char*)malloc(sizeof(char) * 256);
    createViewableImageFile(filestr, img_filestr);
    if(!img_filestr) {
        fprintf(stderr, "Failed to convert pdf to viewable image.\n");
        return 1;
    }
    cv::Mat imageToProcess = cv::imread(img_filestr, CV_LOAD_IMAGE_UNCHANGED);
    unlink(img_filestr);
    if(imageToProcess.empty()) {
        fprintf(stderr, "Failed to load image file %s as cv::Mat\n", filestr);
        return 1;
    }

    createImageWithOverlay(imageToProcess, filestr);

    return 0;
}

int main(int argc, char** argv) {
    if(argc != 2) {
        printf("Incorrect format. Proper usage is: %s <filename.pdf>\n", argv[0]);
        return 1;
    }
    char* filename = argv[1];
    return processImageFile(filename);
}
