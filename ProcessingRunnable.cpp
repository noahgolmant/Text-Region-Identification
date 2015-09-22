/*
 * OrganizerProcessor.cpp
 *
 *  Created on: May 10, 2015
 *      Author: Noah
 */

#include <jni.h>
#include "ProcessingRunnable.h"

/**
 * JNI FUNCTIONS
 */
JNIEXPORT jobject JNICALL Java_application_controller_processor_ProcessingRunnable_getViewableImageFile
        (JNIEnv *, jobject, jstring);
JNIEXPORT jobject JNICALL Java_application_controller_processor_ProcessingRunnable_processImageFile
        (JNIEnv *, jobject, jstring);
/**
 * JNI UTILITY FUNCTIONS
 */



/**
 * JNI link to Java native function.
 * Refer to implementation of getViewableImageFile()
 */
JNIEXPORT jobject JNICALL Java_application_controller_processor_ProcessingRunnable_createViewableImageFile
	(JNIEnv *env, jobject thisobj, jstring filestr) {
    return createViewableImageFile(env, thisobj, filestr);
}


JNIEXPORT jobject JNICALL Java_application_controller_processor_ProcessingRunnable_processImageFile
	(JNIEnv *env, jobject thisobj, jstring filestr) {
    return processImageFile(env, thisobj, filestr);
}


