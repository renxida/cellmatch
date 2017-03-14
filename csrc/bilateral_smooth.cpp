#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;

/// Global variables

Mat src, src_gray;
Mat dst, smoothed;

char* window_name = "Bilateral Filter Smoothing";

int d = 9;
int sigma_color;
int sigma_space;


/**
 * @function CannyThreshold
 * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
 */
void Draw(int, void*)
{
  // apply bilateral filter and show
  bilateralFilter( src_gray, smoothed, d, sigma_color, sigma_space);

  imshow( window_name, smoothed );
 }


/** @function main */
int main( int argc, char** argv )
{
  /// Load an image
  src = imread( argv[1] );

  if( !src.data )
  { return -1; }

  /// Create a matrix of the same type and size as src (for dst)
  dst.create( src.size(), src.type() );

  /// Convert the image to grayscale
  cvtColor( src, src_gray, CV_BGR2GRAY );

  /// Create a window
  namedWindow( window_name, CV_WINDOW_AUTOSIZE );

  /// Create a Trackbar for user to enter threshold
  createTrackbar( "Smoothing Diameter", window_name, &d, 32, Draw );
  createTrackbar( "Sigma de Colour", window_name, &sigma_color, 256, Draw );
  createTrackbar( "Sigma de Space", window_name, &sigma_space, 256, Draw );

  /// Show the image
  Draw(0, 0);

  /// Wait until user exit program by pressing a key
  waitKey(0);

  return 0;
  }
