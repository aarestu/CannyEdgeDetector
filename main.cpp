#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

void canny(const Mat src, Mat& result, int besarKernel, double delta, int thresMax, int thresMin);
void follow(Mat nonmax, Mat &result, int x, int y, int thresMin);
void gaussianKernelGenerator(Mat &resultX, Mat &resultY, Mat &result, int besarKernel, double delta);
void rgb2gray(const Mat src, Mat &result);

int main(int argc, char *argv[])
{
    Mat src = imread("D:\\Project\\C++\\CitraDigital\\shapes.png");

    rgb2gray(src, src);

    Mat edge;

    canny(src, edge, 5, 1.2, 127, 63);

    namedWindow("asli");
    imshow("asli", src);

    namedWindow("CannyEdgeDetector");
    imshow("CannyEdgeDetector", edge);

    waitKey(0);

}

void canny(const Mat src, Mat& result, int besarKernel, double delta, int thresMax, int thresMin)
{
    //inisialisasi kernel
    Mat kernelX;
    Mat kernelY;
    Mat kernelGu;
    gaussianKernelGenerator( kernelX, kernelY, kernelGu, besarKernel, delta);

    int filterOffset = besarKernel / 2;

    Mat mgu =  (Mat_<double>(src.rows - filterOffset * 2, src.cols - filterOffset*2));
    Mat mmagnitude =  (Mat_<double>(mgu.rows - filterOffset * 2, mgu.cols - filterOffset*2));

    Mat mdgux =  (Mat_<double>(mgu.rows - filterOffset * 2, mgu.cols - filterOffset*2));
    Mat mdguy =  (Mat_<double>(mgu.rows - filterOffset * 2, mgu.cols - filterOffset*2));
    Mat dir =  (Mat_<uchar>(mgu.rows - filterOffset * 2, mgu.cols - filterOffset*2));

    Mat nonmax = Mat::zeros(mgu.rows - filterOffset*2, mgu.cols - filterOffset*2, src.type());

    result = Mat::zeros(mgu.rows - filterOffset*2, mgu.cols - filterOffset*2, src.type());

    double sX;
    double sY;
    double sg;
    double magnitude;

    //smooth image with gaussian filter
    for(int ysrc = filterOffset; ysrc < src.rows - filterOffset; ++ysrc){
        for(int xsrc = filterOffset; xsrc < src.cols - filterOffset; ++xsrc){
            sg = 0;
            for(int xkernel = -filterOffset; xkernel <= filterOffset; ++xkernel){
                for(int ykernel = -filterOffset; ykernel <= filterOffset; ++ykernel){
                    sg += src.at<uchar>(ysrc + ykernel, xsrc + xkernel) * kernelGu.at<double>(filterOffset + ykernel, filterOffset + xkernel);
                }
            }
            mgu.at<double>(ysrc - filterOffset, xsrc - filterOffset) = sg;
        }
    }

    double direction;

    //compute derivative of filter image
    for(int ysrc = filterOffset; ysrc < mgu.rows - filterOffset; ++ysrc){
        for(int xsrc = filterOffset; xsrc < mgu.cols - filterOffset; ++xsrc){

            sX = 0;
            sY = 0;
            for(int xkernel = -filterOffset; xkernel <= filterOffset; ++xkernel){
                for(int ykernel = -filterOffset; ykernel <= filterOffset; ++ykernel){
                    sX += mgu.at<double>(ysrc + ykernel, xsrc + xkernel) * kernelX.at<double>(filterOffset + ykernel, filterOffset + xkernel);
                    sY += mgu.at<double>(ysrc + ykernel, xsrc + xkernel) * kernelY.at<double>(filterOffset + ykernel, filterOffset + xkernel);
                }
            }

            mdgux.at<double>(ysrc - filterOffset, xsrc - filterOffset) = sX;
            mdguy.at<double>(ysrc - filterOffset, xsrc - filterOffset) = sY;

            direction = (atan2(sX,sY)/M_PI) * 180.0;		// Calculate actual direction of edge

            if ( ( (direction < 22.5) && (direction > -22.5) ) || (direction > 157.5) || (direction < -157.5) )
                direction = 0;
            if ( ( (direction > 22.5) && (direction < 67.5) ) || ( (direction < -112.5) && (direction > -157.5) ) )
                direction = 45;
            if ( ( (direction > 67.5) && (direction < 112.5) ) || ( (direction < -67.5) && (direction > -112.5) ) )
                direction = 90;
            if ( ( (direction > 112.5) && (direction < 157.5) ) || ( (direction < -22.5) && (direction > -67.5) ) )
                direction = 135;

            dir.at<uchar>(ysrc - filterOffset, xsrc - filterOffset) = direction;

            magnitude = sqrt(pow(sX, 2) + pow(sY, 2));



            mmagnitude.at<double>(ysrc - filterOffset, xsrc - filterOffset) = magnitude;


        }
    }

    for(int y = 1; y < mmagnitude.rows - 1; ++y){
        for(int x = 1; x < mmagnitude.cols - 1; ++x){

            //non-maximal suppression
            switch(dir.at<uchar>(y, x)){
                case 0 :
                    if(mmagnitude.at<double>(y,x) > mmagnitude.at<double>(y + 1,x) && mmagnitude.at<double>(y,x) > mmagnitude.at<double>(y - 1, x)){
                        nonmax.at<uchar>(y,x) = (mmagnitude.at<double>(y,x) <= 255)? mmagnitude.at<double>(y,x) : 255;
                    }else{
                        nonmax.at<uchar>(y,x) = 0;
                    }
                    break;
                case 45 :
                    if(mmagnitude.at<double>(y,x) > mmagnitude.at<double>(y + 1,x + 1) && mmagnitude.at<double>(y,x) > mmagnitude.at<double>(y - 1,x  - 1)){
                        nonmax.at<uchar>(y,x) = (mmagnitude.at<double>(y,x) <= 255)? mmagnitude.at<double>(y,x) : 255;
                    }else{
                        nonmax.at<uchar>(y,x) = 0;
                    }
                    break;
                case 90 :
                    if(mmagnitude.at<double>(y,x) > mmagnitude.at<double>(y,x - 1) && mmagnitude.at<double>(y,x) > mmagnitude.at<double>(y,x  + 1)){
                        nonmax.at<uchar>(y,x) = (mmagnitude.at<double>(y,x) <= 255)? mmagnitude.at<double>(y,x) : 255;
                    }else{
                        nonmax.at<uchar>(y,x) = 0;
                    }
                    break;
                case 135 :
                    if(mmagnitude.at<double>(y,x) > mmagnitude.at<double>(y + 1,x - 1) && mmagnitude.at<double>(y,x) > mmagnitude.at<double>(y - 1,x  + 1)){
                        nonmax.at<uchar>(y,x) = (mmagnitude.at<double>(y,x) <= 255)? mmagnitude.at<double>(y,x) : 255;
                    }else{
                        nonmax.at<uchar>(y,x) = 0;
                    }
                    break;
            }



        }
    }

    //hysteria threshold
    for(int y = 1; y < mmagnitude.rows - 1; ++y){
        for(int x = 1; x < mmagnitude.cols - 1; ++x){
            if(nonmax.at<uchar>(y,x) > thresMax && result.at<uchar>(y,x) < thresMin){
                follow(nonmax, result, x, y, thresMin);
            }
        }
    }

}

void follow(Mat nonmax, Mat &result, int x, int y, int thresMin)
{
    result.at<uchar>(y,x) = nonmax.at<uchar>(y,x);

    if( result.at<uchar>(y + 1,x + 1) < thresMin &&  nonmax.at<uchar>(y + 1,x + 1) > thresMin){
        follow(nonmax, result, x + 1, y + 1, thresMin);
    }
    if( result.at<uchar>(y - 1,x - 1) < thresMin &&  nonmax.at<uchar>(y - 1,x - 1) > thresMin){
        follow(nonmax, result, x - 1, y - 1, thresMin);
    }
    if( result.at<uchar>(y, x - 1) < thresMin &&  nonmax.at<uchar>(y,x - 1) > thresMin){
        follow(nonmax, result, x - 1, y, thresMin);
    }
    if( result.at<uchar>(y, x + 1) < thresMin &&  nonmax.at<uchar>(y,x + 1) > thresMin){
        follow(nonmax, result, x + 1, y, thresMin);
    }
    if( result.at<uchar>(y - 1, x) < thresMin &&  nonmax.at<uchar>(y - 1,x) > thresMin){
        follow(nonmax, result, x, y - 1, thresMin);
    }
    if( result.at<uchar>(y + 1, x) < thresMin &&  nonmax.at<uchar>(y + 1,x) > thresMin){
        follow(nonmax, result, x, y + 1, thresMin);
    }
    if( result.at<uchar>(y + 1, x - 1) < thresMin &&  nonmax.at<uchar>(y + 1, x - 1) > thresMin){
        follow(nonmax, result, x - 1, y + 1, thresMin);
    }
    if( result.at<uchar>(y - 1, x + 1) < thresMin &&  nonmax.at<uchar>(y - 1, x + 1) > thresMin){
        follow(nonmax, result, x + 1, y - 1, thresMin);
    }
}

//Gaussian
void gaussianKernelGenerator(Mat &resultX, Mat &resultY, Mat &result, int besarKernel, double delta)
{
    int kernelRadius = besarKernel / 2;
    resultX = Mat_<double>(besarKernel, besarKernel);
    resultY = Mat_<double>(besarKernel, besarKernel);
    result  = Mat_<double>(besarKernel, besarKernel);

    double pengali = -1 / ( sqrt(2 * (22 / 7) ) * pow(delta, 3) ) ;
    double pengaliGu =  1 / (  sqrt(2 * (22 / 7)) * delta ) ;

    for(int filterX = - kernelRadius; filterX <= kernelRadius; filterX++){
        for(int filterY = - kernelRadius; filterY <= kernelRadius; filterY++){

            resultX.at<double>(filterY + kernelRadius, filterX + kernelRadius) =
                    exp(-( ( pow(filterX, 2)  ) / ( pow(delta, 2) * 2) ))
                    * pengali *filterX;

            resultY.at<double>(filterY + kernelRadius, filterX + kernelRadius) =
                    exp(-( ( pow(filterY, 2)  ) / ( pow(delta, 2) * 2) ))
                    * pengali * filterY;

            result.at<double>(filterY + kernelRadius, filterX + kernelRadius) =
                    exp(-( ( pow(filterY, 2)  ) / ( pow(delta, 2) * 2) ))
                    * pengaliGu;
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
