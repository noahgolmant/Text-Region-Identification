//
// Created by noah on 7/6/15.
//
#include <ctype.h>
#include <ghostscript/iapi.h>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>
#include "ProcessingRunnable.h"

/*
 * STATIC DATA
 */
static char* conversion_args[8];
static void* gs_instance;
static char* numpages_args[3];
static bool receivedPageCount = false;

static inline char* formatArgs(int, char **);
static int GSDLLCALL gs_stdin(void *, char *, int);
static int GSDLLCALL gs_stderr(void *, const char *, int);
static int GSDLLCALL gs_stdout(void *, const char *, int);

/*
 * EXPORTED DATA
 */
int pageCount = 0;
int convertPDF(const char*, const char*);

/**
 * Formats arguments sent to ghostscript for error processing
 */
static inline char* formatArgs(int argc, char** argv) {
    // allocate the correct space for the display buffer
    char *out = (char*)malloc(sizeof(char) * 2048);

    int i;
    for(i = 0; i < argc; i++) {
        char* buf = (char*) malloc(sizeof(argv[i]) + 6);
        sprintf(buf, "    %s\n", argv[i]);
        strcat(out, buf);
    }
    return out;
}

/* stdio functions to read ghostscript output */
static int GSDLLCALL
gs_stdin(void *instance, char *buf, int len)
{
    int ch;
    int count = 0;
    while (count < len) {
        ch = fgetc(stdin);
        if (ch == EOF)
            return 0;
        *buf++ = ch;
        count++;
        if (ch == '\n')
            break;
    }
    return count;
}

static int GSDLLCALL
gs_stdout(void *instance, const char *str, int len)
{
    // the first output of a single digit is
    // the pagecount from the first ghostscript
    // argument list
    if(isdigit(str[0]) && !receivedPageCount) {
        pageCount = atoi(str);
        receivedPageCount = true;
    }
    /* we really don't care about output */
    //fwrite(str, 1, len, stdout);
    //fflush(stdout);
    return len;
}

static int GSDLLCALL
gs_stderr(void *instance, const char *str, int len)
{
    fwrite(str, 1, len, stderr);
    fflush(stderr);
    return len;
}

/**
 * Converts a given PDF to a series of images to be concatenated.
 */
int convertPDF(const char* in, const char* out) {

    int retCode;

    // initialize the ghostscript instance
    retCode = gsapi_new_instance(&gs_instance, NULL);
    if (retCode < 0) {
        fprintf(stderr, "Failed to initialize ghostscript instance.\n");
        fflush(stderr);
        gsapi_delete_instance(gs_instance);
        return GS_FAILURE;
    }

    // Set the IO callback functions to replace stdin, stdout, stderr
    // so we can get the page number count and handle error data explicitly
    gsapi_set_stdio(gs_instance, gs_stdin, gs_stdout, gs_stderr);

    // set up argument text encoding
    retCode = gsapi_set_arg_encoding(gs_instance, GS_ARG_ENCODING_UTF8);
    if(retCode != 0) {
        fprintf(stderr, "Failed to set ghoscript argument text encoding.\n");
        fflush(stderr);
        gsapi_delete_instance(gs_instance);
        return GS_FAILURE;
    }

    //-----------------------------------------------------
    // Construct the command to count the number of pages

    // First, construct the base arguments of gs
    numpages_args[0] = strdup("countPages");
    numpages_args[1] = strdup("-q");
    numpages_args[2] = strdup("-dNODISPLAY");

    // start argument processing for page count
    retCode = gsapi_init_with_args(gs_instance, 3, numpages_args);

    if(retCode != 0 && retCode != -101 /* e_Quit */) {
        fprintf(stderr,
                "Failed to initialize ghostscript pdf pagecount with arguments: \n%s",
                formatArgs(3, numpages_args));
        fflush(stderr);
        gsapi_delete_instance(gs_instance);
        return GS_FAILURE;
    }

    // construct the postscript command to actually count it
    //(input.pdf) (r) file runpdfbegin pdfpagecount = quit
    const char* base_cmd = "(%s) (r) file runpdfbegin pdfpagecount == flush";
    char* command = (char*)malloc(
            snprintf(NULL, 0,base_cmd, in)+1
    );
    sprintf(command, base_cmd, in);
    //gsapi_run_string_begin(gs_instance, 0, &retCode);
    gsapi_run_string(gs_instance, command, 0, &retCode);

    //------------------------------------------------------------------------------

    //------------------------------------------------------------------------------
    // Construct the ghostscript command to create the JPEG
    conversion_args[0] = strdup("convertPDF");
    conversion_args[1] = strdup("-dNOPAUSE");
    conversion_args[2] = strdup("-sDEVICE=bmp256");
    conversion_args[3] = strdup("-r275");
    conversion_args[4] = strdup("-dFirstPage=1");
    // construct last page string
    const char* last_page_base_cmd = "-dLastPage=%d";
    conversion_args[5] = (char*)malloc(
            snprintf(NULL, 0,last_page_base_cmd, pageCount)+1
    );
    sprintf(conversion_args[5], last_page_base_cmd, pageCount);

    // construct output file string
    const char* output_file_base_cmd = "-sOutputFile=%%d%s";
    conversion_args[6] = (char*)malloc(snprintf(NULL, 0 , output_file_base_cmd, out)+1);
    sprintf(conversion_args[6], output_file_base_cmd, out);
    conversion_args[7] = strdup(in);

    //num_pages_args[2] = strdup("-dNODISPLAY");

    // start argument processing for pdf conversion
    retCode = gsapi_init_with_args(gs_instance, 8, conversion_args);

    if(retCode != 0 && retCode != -101 /* e_Quit */) {
        fprintf(stderr,
                "Failed to initialize ghostscript pdf conversion with arguments: \n%s",
                formatArgs(8, conversion_args));
        fflush(stderr);
        gsapi_delete_instance(gs_instance);
        return GS_FAILURE;
    }
    //--------------------------------------------------------------------------------

    retCode = gsapi_exit(gs_instance);
    if(retCode != 0) {
        fprintf(stderr, "Failed to exit ghostscript correctly.\n");
        fflush(stderr);
        gsapi_delete_instance(gs_instance);
        return GS_FAILURE;
    }

    gsapi_delete_instance(gs_instance);
    return GS_SUCCESS;
}
