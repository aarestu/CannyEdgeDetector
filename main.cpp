#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

void canny(const Mat src, Mat& result, int thresMax, int thresMin, Mat kernelX, Mat kernelY);
void follow(Mat nonmax, Mat &result, int x, int y, int thresMin, int thresMax);
void gaussianKernelDerivativeGenerator(Mat &resultX, Mat &resultY, int besarKernel, double delta);
void rgb2gray(const Mat src, Mat &result);

int main(int argc, char *argv[])
{
    Mat src = imread("D:\\Project\\C++\\CitraDigital\\shapes.png");

    VideoCapture cap(1);
    if (!cap.isOpened())
    {
        cout << "ga bisa buka video cam" << endl;
        return -1;
    }

    namedWindow("CannyEdgeDetector");

    //inisialisasi kernel
    Mat dGx;
    Mat dGy;
    gaussianKernelDerivativeGenerator(dGx, dGy, 3, 0.6);
    cout<<"dGx :"<<endl<<dGx<<endl<<endl;
    cout<<"dGy :"<<endl<<dGy<<endl<<endl;

    while(1){
        if(cap.read(src)){
            rgb2gray(src, src);

            Mat edge;
            namedWindow("asli");
            imshow("asli", src);

            canny(src, edge, 19, 6, dGx, dGy);




            imshow("CannyEdgeDetector", edge);
        }
        if (waitKey(1) == 27)
        {
           break;
        }
    }
}

void canny(const Mat src, Mat& result, int thresMax, int thresMin, Mat kernelX, Mat kernelY)
{
    int centerKernel = kernelX.cols / 2;
    int cols = src.cols;
    int rows = src.rows;

    Mat mmagnitude =  (Mat(src.rows, src.cols,CV_64FC1, Scalar(0)));

    Mat direction =  (Mat(src.rows, src.cols,CV_64FC1, Scalar(0)));


    Mat nonmax  =  (Mat(src.rows, src.cols,CV_8UC1, Scalar(0)));

    result = Mat(src.rows, src.cols,CV_8UC1, Scalar(0));

    double sX, sY;

    int ii, jj;


    //compute derivative of filter image

    for(int i = 0; i < cols; ++i){
        for(int j = 0; j < rows; ++j){
            sX = 0;
            sY = 0;

            for(int ik = -centerKernel; ik <= centerKernel; ++ik ){
                ii = i + ik;
                for(int jk = -centerKernel; jk <= centerKernel; ++jk ){
                    jj = j + jk;

                    if(ii >= 0 && ii < cols && jj >= 0 && jj < rows){
                        sX += src.at<uchar>(jj, ii) * kernelX.at<double>(centerKernel + jk, centerKernel + ik);
                        sY += src.at<uchar>(jj, ii) * kernelY.at<double>(centerKernel + jk, centerKernel + ik);
                    }
                }
            }

            direction.at<double>(j, i) = (atan2(sX, sY)/M_PI) * 180.0;
            mmagnitude.at<double>(j, i) = sqrt(pow(sX, 2) + pow(sY, 2));
        }
    }

    for(int y = 1; y < rows - 1; ++y){
        for(int x = 1; x < cols - 1; ++x){

            //non-maximal suppression


            if ( ( (direction.at<double>(y, x) < 22.5) && (direction.at<double>(y, x) > -22.5) ) || (direction.at<double>(y, x) > 157.5) || (direction.at<double>(y, x) < -157.5) ){
                if(mmagnitude.at<double>(y,x) > mmagnitude.at<double>(y + 1,x) && mmagnitude.at<double>(y,x) > mmagnitude.at<double>(y - 1, x)){
                    if(mmagnitude.at<double>(y,x) > thresMax){
                        result.at<uchar>(y,x) = 255;
                    }

                    else if(mmagnitude.at<double>(y,x) <= thresMax && mmagnitude.at<double>(y,x) > thresMin){
                        nonmax.at<uchar>(y,x) = mmagnitude.at<double>(y,x);
                    }
                }
            }
            if ( ( (direction.at<double>(y, x) > 22.5) && (direction.at<double>(y, x) < 67.5) ) || ( (direction.at<double>(y, x) < -112.5) && (direction.at<double>(y, x) > -157.5) ) ){
                if(mmagnitude.at<double>(y,x) > mmagnitude.at<double>(y + 1,x + 1) && mmagnitude.at<double>(y,x) > mmagnitude.at<double>(y - 1,x  - 1)){
                    if(mmagnitude.at<double>(y,x) > thresMax){
                        result.at<uchar>(y,x) = 255;
                    }

                    else if(mmagnitude.at<double>(y,x) <= thresMax && mmagnitude.at<double>(y,x) > thresMin){
                        nonmax.at<uchar>(y,x) = mmagnitude.at<double>(y,x);
                    }
                }
            }
            if ( ( (direction.at<double>(y, x) > 67.5) && (direction.at<double>(y, x) < 112.5) ) || ( (direction.at<double>(y, x) < -67.5) && (direction.at<double>(y, x) > -112.5) ) ){
                if(mmagnitude.at<double>(y,x) > mmagnitude.at<double>(y,x - 1) && mmagnitude.at<double>(y,x) > mmagnitude.at<double>(y,x  + 1)){
                    if(mmagnitude.at<double>(y,x) > thresMax){
                        result.at<uchar>(y,x) = 255;
                    }

                    else if(mmagnitude.at<double>(y,x) <= thresMax && mmagnitude.at<double>(y,x) > thresMin){
                        nonmax.at<uchar>(y,x) = mmagnitude.at<double>(y,x);
                    }
                }
            }
            if ( ( (direction.at<double>(y, x) > 112.5) && (direction.at<double>(y, x) < 157.5) ) || ( (direction.at<double>(y, x) < -22.5) && (direction.at<double>(y, x) > -67.5) ) ){
                if(mmagnitude.at<double>(y,x) > mmagnitude.at<double>(y + 1,x - 1) && mmagnitude.at<double>(y,x) > mmagnitude.at<double>(y - 1,x  + 1)){

                    if(mmagnitude.at<double>(y,x) > thresMax){
                        result.at<uchar>(y,x) = 255;
                    }

                    else if(mmagnitude.at<double>(y,x) <= thresMax && mmagnitude.at<double>(y,x) > thresMin){
                        nonmax.at<uchar>(y,x) = mmagnitude.at<double>(y,x);
                    }
                }
            }
        }
    }

    //hysteria threshold
    for(int y = 1; y < rows - 1; ++y){
        for(int x = 1; x < cols - 1; ++x){
            if(result.at<uchar>(y,x) == 255){
                follow(nonmax, result, x, y, thresMin, thresMax);
            }
        }
    }
}


void follow(Mat nonmax, Mat &result, int x, int y, int thresMin, int thresMax)
{
    result.at<uchar>(y,x) = 255;

    if( result.at<uchar>(y + 1,x + 1) == 0 &&  nonmax.at<uchar>(y + 1,x + 1) > thresMin && nonmax.at<uchar>(y + 1,x + 1) <= thresMax){
        follow(nonmax, result, x + 1, y + 1, thresMin, thresMax);
    }
    if( result.at<uchar>(y - 1,x - 1) == 0 &&  nonmax.at<uchar>(y - 1,x - 1) > thresMin && nonmax.at<uchar>(y + 1,x + 1) <= thresMax){
        follow(nonmax, result, x - 1, y - 1, thresMin, thresMax);
    }
    if( result.at<uchar>(y, x - 1) == 0 &&  nonmax.at<uchar>(y,x - 1) > thresMin && nonmax.at<uchar>(y + 1,x + 1) <= thresMax){
        follow(nonmax, result, x - 1, y, thresMin, thresMax);
    }
    if( result.at<uchar>(y, x + 1) == 0 &&  nonmax.at<uchar>(y,x + 1) > thresMin && nonmax.at<uchar>(y + 1,x + 1) <= thresMax){
        follow(nonmax, result, x + 1, y, thresMin, thresMax);
    }
    if( result.at<uchar>(y - 1, x) == 0 &&  nonmax.at<uchar>(y - 1,x) > thresMin && nonmax.at<uchar>(y + 1,x + 1) <= thresMax){
        follow(nonmax, result, x, y - 1, thresMin, thresMax);
    }
    if( result.at<uchar>(y + 1, x) == 0 &&  nonmax.at<uchar>(y + 1,x) > thresMin && nonmax.at<uchar>(y + 1,x + 1) <= thresMax){
        follow(nonmax, result, x, y + 1, thresMin, thresMax);
    }
    if( result.at<uchar>(y + 1, x - 1) == 0 &&  nonmax.at<uchar>(y + 1, x - 1) > thresMin && nonmax.at<uchar>(y + 1,x + 1) <= thresMax){
        follow(nonmax, result, x - 1, y + 1, thresMin, thresMax);
    }
    if( result.at<uchar>(y - 1, x + 1) == 0 &&  nonmax.at<uchar>(y - 1, x + 1) > thresMin && nonmax.at<uchar>(y + 1,x + 1) <= thresMax){
        follow(nonmax, result, x + 1, y - 1, thresMin, thresMax);
    }
}

//Gaussian
void gaussianKernelDerivativeGenerator(Mat &resultX, Mat &resultY, int besarKernel, double delta)
{
    int kernelRadius = besarKernel / 2;
    resultX = Mat_<double>(besarKernel, besarKernel);
    resultY = Mat_<double>(besarKernel, besarKernel);

    double pengali = -1 / ( 2 * (22 / 7) * pow(delta, 4) ) ;

    for(int filterX = - kernelRadius; filterX <= kernelRadius; filterX++){
        for(int filterY = - kernelRadius; filterY <= kernelRadius; filterY++){

            resultX.at<double>(filterY + kernelRadius, filterX + kernelRadius) =
                    exp(-( ( pow(filterX, 2) + pow(filterY, 2)  ) / ( pow(delta, 2) * 2) ))
                    * pengali * filterX;

            resultY.at<double>(filterY + kernelRadius, filterX + kernelRadius) =
                    exp(-( ( pow(filterX, 2) + pow(filterY, 2) ) / ( pow(delta, 2) * 2) ))
                    * pengali * filterY;

        }

    }

    //cout<< result << endl;
    //cout<< resultY << endl;
}


void rgb2gray(const Mat src, Mat &result)
{
    CV_Assert(src.depth() != sizeof(uchar)); //harus 8 bit

    result = Mat::zeros(src.rows, src.cols, CV_8UC1); //buat matrik 1 chanel
    uchar data;

    if(src.channels() == 3){

        for( int i = 0; i < src.rows; ++i)
            for( int j = 0; j < src.cols; ++j )
            {
                data = (uchar)(((Mat_<Vec3b>) src)(i,j)[0] * 0.0722 + ((Mat_<Vec3b>) src)(i,j)[1] * 0.7152 + ((Mat_<Vec3b>) src)(i,j)[2] * 0.2126);

                result.at<uchar>(i,j) = data;
            }


    }else{

        result = src;
    }

}
