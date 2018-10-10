#include <jni.h>
#include <string>
#include <opencv2/opencv.hpp>
#include "CustomCodeRecognition.h"
using namespace cv;
const int Cut_Szie = 1024;

extern "C"
JNIEXPORT jstring JNICALL
Java_customcode_customcode_1android_CustomCodeActivity_CustomCode(JNIEnv *env, jobject instance,
                                                            jlong matAddrInput,
                                                            jlong matAddrResult) {

    // TODO
    Mat &matInput = *(Mat *)matAddrInput;
    Mat &matResult = *(Mat *)matAddrResult;
    cvtColor(matInput, matInput, CV_BGRA2BGR);
    Mat input = matInput.clone();
    Mat input_cut;
    if(input.cols < Cut_Szie || input.rows < Cut_Szie)
    {
        if(input.cols < Cut_Szie/2 || input.rows < Cut_Szie/2) return env->NewStringUTF(0);
        int MinSize = (input.cols < input.rows)? input.cols : input.rows;
        input_cut = input(Rect(input.cols / 2 - MinSize / 2,
                            input.rows / 2 - MinSize / 2, MinSize,
                            MinSize));
    }
    else {
        input_cut = input(Rect(input.cols / 2 - Cut_Szie / 2,
                            input.rows / 2 - Cut_Szie/ 2, Cut_Szie, Cut_Szie));

    }

    string result = "";
    CustomCode* customcode = new CustomCode();
    vector<Point2f> markers;
    customcode->recognition(&matInput, &markers, &result);
    delete  customcode;
    return env->NewStringUTF(result.c_str());
}

extern "C"
JNIEXPORT void JNICALL
Java_customcode_customcode_1android_CustomCodeActivity_DrawRecognitionArea(JNIEnv *env, jobject instance,
                                                                                jlong matAddrInput,
                                                                                jlong matAddrResult) {

    // TODO
    Mat &matInput = *(Mat*)matAddrInput;
    Mat &matResult = *(Mat*)matAddrResult;
    if(matInput.cols == 0 || matInput.rows == 0) return;
    Mat input = matInput.clone();
    if(input.cols >= Cut_Szie && input.rows >= Cut_Szie) {
        rectangle(input, Rect(input.cols / 2 - Cut_Szie / 2,
                              input.rows / 2 - Cut_Szie / 2, Cut_Szie,
                              Cut_Szie), Scalar(255, 0, 0), 5);
    } else{
        if(input.cols == 0 || input.rows == 0) return;
        int MinSize = ((input.cols < input.rows)? input.cols : input.rows);
        rectangle(input, Rect(input.cols / 2 - MinSize / 2,
                              input.rows / 2 - MinSize / 2, MinSize,
                              MinSize), Scalar(255, 0, 0), 1);
    }
    cvtColor(input, input, CV_BGRA2BGR);
    matResult = input;
//    matInput = input;

}
