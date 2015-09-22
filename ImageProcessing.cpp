//
// Created by noah on 7/15/15.
//
#include <jni.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <sys/stat.h>
#include <sys/time.h>
#include <tesseract/baseapi.h>
#include <unistd.h>
#include "ProcessingRunnable.h"

/* UTILITY FUNCTIONS */
static inline int max(int, int);
static inline bool fileExists(char *);

typedef struct {
    float x,y,width,height;
} normalizedRect;

static inline normalizedRect getNormalizedRect(cv::Mat, cv::Rect);

/**
 * OPENCV FUNCTIONS
 */
jobject createViewableImageFile(JNIEnv *, jobject, jstring);
jobject processImageFile(JNIEnv *, jobject, jstring);

/* JNI PROGRESS CALBACK */
static void updateProgress(JNIEnv*, jobject, jdouble);

static std::vector<cv::Rect> getTextRegions(JNIEnv*, jobject, cv::Mat);
static std::vector<char*>    getTextForRegions(JNIEnv*, jobject, cv::Mat, std::vector<cv::Rect>);

/** PROGRESS CONSTANTS */
static const jdouble CREATE_IMAGE_PROGRESS     = 0.15;
static const jdouble SOBEL_PROGRESS            = 0.20;
static const jdouble MORPH_PROGRESS            = 0.225;
static const jdouble CONTOUR_PROGRESS          = 0.25;
static const jdouble GROUP_RECTANGLES_PROGRESS = 0.30;
static const jdouble TEXT_PROGRESS_LEFT        = 1.0 - GROUP_RECTANGLES_PROGRESS;

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

static inline float round_digits(float f) {
    float t = powf(10.0, NUM_COORD_DIGITS);
    return roundf(f * t) / t;
}

static inline normalizedRect getNormalizedRect(cv::Mat mat, cv::Rect rect) {
    float x      = round_digits(((float)rect.x) / mat.cols);
    float y      = round_digits(((float)rect.y) / mat.rows);
    float width  = round_digits(((float)rect.width) / mat.cols);
    float height = round_digits(((float)rect.height) / mat.rows);
    return {x,y,width,height};
}

static inline long getTimeMilliseconds() {
    timeval tv;
    gettimeofday(&tv, NULL);

    return tv.tv_usec / 1000;
}

static void updateProgress(JNIEnv* env, jobject thisobj, jdouble newVal) {
    jclass jProcessingRunnable = env->FindClass("application/controller/processor/ProcessingRunnable");
    jmethodID jUpdateProgress = env->GetMethodID(jProcessingRunnable, "updateProgress", "(D)V");
    env->CallObjectMethod(thisobj, jUpdateProgress, newVal);
}


/**
 * JNI function to retrieve access to the viewable image file that contains
 * the concatenation of the pages of the original document.
 *
 * If the file already exists, it returns that. Otherwise, it creates a new one
 * with a String input of the absolute path of the document.
 *
 * @param filestr String representation of the absolute path of the document
 */
jobject createViewableImageFile(JNIEnv *env, jobject thisobj, jstring filestr) {
    // convert java filename string to c filename str
    const char *cfilestr = env->GetStringUTFChars(filestr,0);

    // call the ghostscript conversion process to
    // create separate images per page of the document
    int convertRet = convertPDF(cfilestr, "temp.bmp");
    if(convertRet != GS_SUCCESS) {
        fprintf(stderr, "Failed to convert PDF.\n");
        fflush(stderr);
        env->ReleaseStringUTFChars(filestr, cfilestr);
        return NULL;
    }

    // check to make sure there are actual pages to process
    if(pageCount == 0 /* default value of pageCount */) {
        fprintf(stderr, "No pages to process.\n");
        fflush(stderr);
        env->ReleaseStringUTFChars(filestr, cfilestr);
        return NULL;
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
            fflush(stderr);
            env->ReleaseStringUTFChars(filestr, cfilestr);
            return NULL;
        }

        // store the page's image in the page image array (loading it into memory)
        pageMats[i - 1] = cv::imread(formattedImgStr, CV_LOAD_IMAGE_UNCHANGED);
        //cv::bitwise_not(pageMats[i-1], pageMats[i-1]);
        // check if the OpenCV failed to read the image for some reason
        if(pageMats[i-1].empty()) {
            fprintf(stderr, "Failed to create cv::Mat for %s\n", formattedImgStr);
            fflush(stderr);
            env->ReleaseStringUTFChars(filestr, cfilestr);
            return NULL;
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
    char concatenatedImgFileStr[256];
    strcpy(concatenatedImgFileStr, cfilestr);
    strcat(concatenatedImgFileStr, ".bmp");

    // Write the file from memory
    cv::imwrite(concatenatedImgFileStr, concatenatedImg);

    // Load the java File class and its constructor
    jclass fileClass = env->FindClass("java/io/File");
    if(fileClass == NULL) {
        fprintf(stderr, "Failed to load File class\n");
        fflush(stderr);
        env->ReleaseStringUTFChars(filestr, cfilestr);
        return NULL;
    }                                                            /* File(String) - returns void */
    jmethodID fileConstruct = env->GetMethodID(fileClass, "<init>", "(Ljava/lang/String;)V");
    if(fileConstruct == NULL) {
        fprintf(stderr, "Failed to load File constructor (L)V\n");
        fflush(stderr);
        env->ReleaseStringUTFChars(filestr, cfilestr);
        return NULL;
    }

    // Create a jstring to instantiate the file object with a Java string
    jstring imgFileJStr = env->NewStringUTF(concatenatedImgFileStr);

    // Create the actual image File object
    jobject viewableImageFile = env->NewObject(fileClass, fileConstruct, imgFileJStr);
    if(viewableImageFile == NULL) {
        fprintf(stderr, "Failed to instantiate viewable file object\n");
        fflush(stderr);
    }

    // Garbage collection
    env->ReleaseStringUTFChars(filestr, cfilestr);

    updateProgress(env, thisobj, CREATE_IMAGE_PROGRESS);

    // return to java control
    return viewableImageFile;
}

static std::vector<cv::Rect> getTextRegions(JNIEnv* env, jobject thisobj, cv::Mat imageToProcess /* unedited image */) {
    cv::Mat processingMat(imageToProcess);

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

    updateProgress(env, thisobj, SOBEL_PROGRESS);

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

    updateProgress(env, thisobj, MORPH_PROGRESS);

    #ifdef DEBUG
    time = getTimeMilliseconds() - startTime;
    fprintf(stdout, "--- MORPHOLOGICAL OPERATIONS TIME: %ldms ---\n", abs(time));
    fflush(stdout);
    startTime = getTimeMilliseconds();
    #endif

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i>              hierarchy;

    cv::findContours(processingMat, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

    updateProgress(env, thisobj, CONTOUR_PROGRESS);

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

        /*if((float)contRect.height / imageToProcess.rows  >= 0.0075) {
            // add two of each so even if it's really a single rect, groupRectangles() keeps it
            //fprintf(stdout, "HEIGHT: %f\n", ((float)contRect.height / imageToProcess.rows));
            // fprintf(stderr, "NUM: %f\n", (((float)contRect.height / imageToProcess.rows) / .0075);
            float rowsForR = ((float)contRect.height / imageToProcess.rows) / 0.0075;
            int numRects = (int)(rowsForR + 0.5f);

            //fprintf(stderr, "NUM RECTS: %d\n", numRects);
            //fflush(stderr);
            //fflush(stdout);
            cv::Rect r;

            int height = (int)(((double)contRect.height) / rowsForR);
            int width = contRect.width;
            int x = contRect.x;
            for(int i = 0; i < numRects; i++) {

                int y = (int)(contRect.y + (i * ((double)contRect.height/rowsForR)));


                fprintf(stderr, "%d,%d,%d,%d\n", x, y, width, height);
                fflush(stderr);
                r = cv::Rect(x,y,width,height);

                rectangles.push_back(r);
                rectangles.push_back(r);
            }

        } else {*/
            rectangles.push_back(contRect);
            rectangles.push_back(contRect);
        //}



    }



    /* 1 = group rectangles that have at least one close enough in the cluster
    .*     where the cluster size is determined by epsilon (0.2)
     */
    cv::groupRectangles(rectangles, 1, 0.05);

    updateProgress(env, thisobj, GROUP_RECTANGLES_PROGRESS);

    #ifdef DEBUG
    fprintf(stdout, "--- GROUP RECTANGLES TIME: %ldms ---\n", abs(time));
    fflush(stdout);
    #endif

    /* RECTANGLES ARE SORTED IN DESCENDING Y VALUE (highest -> lowest) */

    // combine rectangles that are close enough on the x-axis
    // i is the current rect index while j is the comparator index
    // while(rectangles[i].y - rectangles[j].y < 5)
    //      if(rectangles[j].x - (rectangles[i].x + rectangles[i].width) < x_dist_threshhold)
    //          combineRects() -- adjust i and j accordingly
    /*int i,j;
    for(i = 0; i < rectangles.size() - 1; i++) {
        j = i + 1;
        while(rectangles[i].y - rectangles[j].y < 25) {
            if(abs(rectangles[j].x - (rectangles[i].x + rectangles[i].width)) < 100) {
                // new width is the gap ((i.x + i.width) - j.x) and j.width
                rectangles[i].width += abs((rectangles[i].x + rectangles[i].width) - rectangles[j].x) + rectangles[j].width;
                rectangles.erase(rectangles.begin() + j);
            } else {
                j++;
            }
        }
    }*/

    return rectangles;
}

static std::vector<char*> getTextForRegions(JNIEnv* env, jobject thisobj, cv::Mat image, std::vector<cv::Rect> regions) {
    std::vector<char*> text;
    cv::cvtColor(image, image, CV_8UC1);
    cv::cvtColor(image, image, CV_RGB2GRAY);

    cv::threshold(image, image, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

    tesseract::TessBaseAPI *api = new tesseract::TessBaseAPI();
    // Initialize tesseract-ocr with English, without specifying tessdata path
    if (api->Init(".", "eng")) {
        fprintf(stderr, "Could not initialize tesseract.\n");
        fflush(stderr);
        return text;
    }

    api->SetImage((uchar*)image.data, image.cols, image.rows, 1, image.cols);

    #ifdef DEBUG
    long totalTime = 0;

    long startTime = getTimeMilliseconds();
    long delta = 0;

    fprintf(stdout, "NUM RECTS: %d\n", regions.size());


    fprintf(stdout, "IMAGE BOUNDS: { %d , %d }\n", image.cols, image.rows);
    fflush(stdout);
    #endif

    double currentProgress = GROUP_RECTANGLES_PROGRESS; /* last update performed */
    double progressPerRegion = TEXT_PROGRESS_LEFT / regions.size();


    for(int i = 0; i < regions.size(); i++) {

        cv::Rect r = regions.at(i);

        #ifdef DEBUG
            //fprintf(stdout,"RECT: %d\n", i);
            //fprintf(stdout, "{ %d , %d , %d , %d }\n", r.x, r.y, r.width, r.height);
            //fflush(stdout);
        #endif

        api->SetRectangle(r.x, r.y, r.width, r.height);
        char* rText = api->GetUTF8Text();
        if(rText && rText != 0x0)
            text.push_back(rText);

        currentProgress += progressPerRegion;
        updateProgress(env, thisobj, currentProgress);

        #ifdef DEBUG
        delta = abs(getTimeMilliseconds() - startTime);
        startTime = getTimeMilliseconds();

        totalTime += delta;
        #endif

        //printf("%s\n", api->GetUTF8Text());
        //fflush(stdout);
    }

    #ifdef DEBUG
    fprintf(stdout, "--- TOTAL TEXT EXTRACTION TIME: %ldms ---\n", totalTime);
    fprintf(stdout, "--- AVG TEXT RECT EXTRACTION TIME: %fms ---\n", ((float)totalTime / regions.size()));
    fflush(stdout);
    #endif

    return text;

}

jobject processImageFile(JNIEnv *env, jobject thisobj, jstring filestr) {

    // check to see if the file actually exists, i.e. it's already been processed
    const char* cfilestr = env->GetStringUTFChars(filestr, 0);
    if(!fileExists(strdup(cfilestr))) {
        fprintf(stderr, "Tried to process an image file that doesn't exist: %s\n", cfilestr);
        fflush(stderr);
        return NULL;
    }

    // initialize hashmap class
    jclass mapClass = env->FindClass("java/util/HashMap");

    if(mapClass == NULL) {
        fprintf(stderr, "Failed to load HashMap class\n");
        fflush(stderr);
        return NULL;
    }

    // store the default constructor method and create a new hashmap object
    jmethodID mapInit = env->GetMethodID(mapClass, "<init>", "()V");
    jobject hashMap = env->NewObject(mapClass, mapInit);

    // create reference to the .put(Object, Object) method
    jmethodID put =env->GetMethodID(mapClass, "put",
                                    "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");


    // initialize Rectangle2D.Double class
    jclass rectClass = env->FindClass("java/awt/geom/Rectangle2D$Double");
    if(rectClass == NULL) {
        fprintf(stderr, "Failed to load Rectangle2D.Double class\n");
        fflush(stderr);
        return NULL;
    }

    // reference constructor with double arguments
    jmethodID rectConstructor = env->GetMethodID(rectClass, "<init>", "(DDDD)V");
    if (rectConstructor == NULL) {
        fprintf(stderr, "Failed to load Rectangle2D.Double constructor (DDDD)V\n");
        fflush(stderr);
        return NULL;
    }

    cv::Mat imageToProcess = cv::imread(cfilestr, CV_LOAD_IMAGE_UNCHANGED);
    if(imageToProcess.empty()) {
        fprintf(stderr, "Failed to load image file %s as cv::Mat\n", cfilestr);
        fflush(stderr);
        return NULL;
    }

    std::vector<cv::Rect> textRegions = getTextRegions(env, thisobj, imageToProcess);

    std::vector<char*> text = getTextForRegions(env, thisobj, imageToProcess, textRegions);

    for(int i = 0; i < textRegions.size(); i++) {
        normalizedRect normRect = getNormalizedRect(imageToProcess, textRegions[i]);
        jobject jRect = env->NewObject(rectClass, rectConstructor,
                (jdouble) (normRect.x),
                (jdouble) (normRect.y),
                (jdouble) (normRect.width),
                (jdouble) (normRect.height)
        );

        jstring jtext = env->NewStringUTF(text[i]);

        env->CallObjectMethod(hashMap, put, jRect, jtext);

    }

    return hashMap;
}