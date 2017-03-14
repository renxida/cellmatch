#include<opencv2/highgui/highgui.hpp>
using namespace cv;

int main(int argc, char *argv[])
{

    Mat img = imread(argv[1], 0);
    imshow("opencvtest",img);
    waitKey(0);

    return 0;
}
