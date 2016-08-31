#include<iostream>
#include<highgui.hpp>
#include<core.hpp>
#include<imgproc.hpp>
using namespace std;
using namespace cv;

#define debug if(1)

typedef vector<vector<Point> > vvp;
typedef vector<Point2f> vp;

int errorOccurred  = 0;

struct CCell
{
    bool marked;
    int x,y,w,h;
};

struct COmrResult
{
    CCell Black[25][21];
    CCell White[25][21];
};


double absDistance(Point2f point1, Point2f point2)
{
    return sqrt(pow((point1.x-point2.x),2)+pow((point1.y-point2.y),2));
}

int invertImage(Mat &src, Mat &dst)
{
    dst=src.clone();
    MatIterator_<uchar> it,end;
    for(it=src.begin<uchar>(), end=src.end<uchar>(); it!=end; it++)
    {
        *it=255-(*it);
    }
}

void gammaCorrection(Mat &src, Mat &dst, float gamma)
{
    unsigned char lut[256];
    for (int i=0;i<256;i++)
    {
        lut[i] = saturate_cast<uchar>(pow((float)(i/255.0),gamma)*255.0f);
    }
    dst = src.clone();
    const int channels = dst.channels();
    switch(channels)
    {
    case 1:
        for(MatIterator_<uchar> it = dst.begin<uchar>(),end=dst.end<uchar>(); it!=end; it++)
        {
            *it = lut[(*it)];
        }
        break;
    case 3:
        MatIterator_<Vec3b> it,end;
        for(it=dst.begin<Vec3b>(), end=dst.end<Vec3b>(); it!=end; it++)
        {
            (*it)[0] = lut[((*it)[0])];
            (*it)[1] = lut[((*it)[1])];
            (*it)[2] = lut[((*it)[2])];
        }
        break;
    }
}


Mat drawHistogram(Mat src)
{
    Mat temp = src.clone();
    Mat histogram;

    vector<Mat> splitPlanes;
    split( temp, splitPlanes);

     /// Establish the number of bins
    int histSize = 256;

      /// Set the ranges ( for B,G,R) )
      float range[] = { 0, 256 } ;
      const float* histRange = { range };
      int hist_w = 512; int hist_h = 400;
      int bin_w = cvRound( (double) hist_w/histSize );
      histogram = Mat::zeros(Size(hist_w,hist_w),CV_8UC3);

      bool uniform = true; bool accumulate = false;
      int imChannels = temp.channels();
    if(imChannels == 1)
    {
        Mat hist;
        calcHist( &splitPlanes[0],1,0,Mat(),hist,1,&histSize,&histRange,uniform,accumulate);
        cv::normalize(hist,hist,0,histogram.rows,NORM_MINMAX,-1,Mat());
        for( int i = 1; i < histSize; i++ )
        {
            line( histogram, Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ) ,
                             Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
                             Scalar( 0, 255, 255), 2, 8, 0 );
        }
    }
    if(imChannels == 3)
    {
      Mat b_hist, g_hist, r_hist;

      /// Compute the histograms:
      calcHist( &splitPlanes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
      calcHist( &splitPlanes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
      calcHist( &splitPlanes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

      /// Normalize the result to [ 0, histImage.rows ]
      normalize(b_hist, b_hist, 0, histogram.rows, NORM_MINMAX, -1, Mat() );
      normalize(g_hist, g_hist, 0, histogram.rows, NORM_MINMAX, -1, Mat() );
      normalize(r_hist, r_hist, 0, histogram.rows, NORM_MINMAX, -1, Mat() );

    /// Draw for each channel
    for( int i = 1; i < histSize; i++ )
    {
       line( histogram, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
                        Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
                        Scalar( 255, 0, 0), 2, 8, 0  );
       line( histogram, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
                        Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
                        Scalar( 0, 255, 0), 2, 8, 0  );
       line( histogram, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
                        Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
                        Scalar( 0, 0, 255), 2, 8, 0  );
        }
    }

    return histogram;
}


void extractInformation(Mat omrSheet, CCell information[25][21])
{
    Mat gray,thresholded, box,otsuOutput1,otsuOutput2,otsuOutput;
    Mat omrDup = omrSheet.clone();
    vvp contours;
    double maxArea;
    double offsetx = 0.0358*omrSheet.cols;
    double offsety = 0.0478*omrSheet.rows;
    double stepX = (omrSheet.cols-offsetx+0.0139*omrSheet.cols)/21.0;
    double stepY = (omrSheet.rows-offsety)/25.0;
    medianBlur(omrDup,omrDup,5);
    cvtColor(omrDup,omrDup,CV_BGR2HSV);
    Mat splitted[3];
    split(omrDup,splitted);
//    namedWindow("Sat",CV_WINDOW_NORMAL);
//    imshow("Sat",splitted[2]);
    //gammaCorrection(splitted[1],gray,0.5);
    invertImage(splitted[2],splitted[2]);
    bitwise_or(splitted[1],splitted[2],otsuOutput);
    namedWindow("otsu123",CV_WINDOW_NORMAL);
    imshow("otsu123",otsuOutput);
    threshold(otsuOutput,otsuOutput,0, 255, CV_THRESH_OTSU);

    //threshold(splitted[1],gray,60,255,CV_THRESH_BINARY_INV);
    //namedWindow("Thre",CV_WINDOW_NORMAL);
    //imshow("Thre",gray);
    namedWindow("otsu",CV_WINDOW_NORMAL);
    imshow("otsu",otsuOutput);

//    Mat hist0 = drawHistogram(splitted[0]);
    namedWindow("Hist0",CV_WINDOW_NORMAL);
    namedWindow("Hist1",CV_WINDOW_NORMAL);
    imshow("Hist0",splitted[1]);
//    Mat hist1 = drawHistogram(splitted[1]);
    imshow("Hist1",splitted[2]);
//    Mat hist2 = drawHistogram(splitted[2]);
//    imshow("Hist2",hist2);
//    waitKey(0);
//    return;
    erode(otsuOutput,otsuOutput,getStructuringElement(MORPH_RECT,Size(3,3)));
//    namedWindow("otsue",CV_WINDOW_NORMAL);
//    imshow("otsue",otsuOutput);
    dilate(otsuOutput,otsuOutput,getStructuringElement(MORPH_RECT,Size(7,7)));
//    namedWindow("otsud",CV_WINDOW_NORMAL);
//    imshow("otsud",otsuOutput);
    //erode(otsuOutput,otsuOutput,getStructuringElement(MORPH_RECT,Size(3,3)));
    //namedWindow("otsu",CV_WINDOW_NORMAL);
    //imshow("otsu",otsuOutput);
    for(int j=0;j<25;j++)
    {
        for(int i=0; i<21; i++)
        {

            Point p2 = Point(offsetx+(i+1)*stepX-offsetx/8,offsety+(j+1)*stepY-0.0047*omrSheet.rows);
            if(p2.x>=omrSheet.cols)
            {
                p2.x=omrSheet.cols - 1;
            }
            if(p2.y>=omrSheet.rows)
            {
                p2.y = omrSheet.rows-1;
            }
            Point p1 = Point(offsetx+i*stepX-offsetx/8,offsety+j*stepY+0.0047*omrSheet.rows);
            if(j==0)
            {
                p1 = Point(offsetx+i*stepX-offsetx/8,offsety+j*stepY+0.0047*omrSheet.rows+offsety/9);
            }
            Rect crop = Rect(p1,p2);
            debug rectangle(omrSheet,Rect(p1,p2),Scalar(0,255,0));
            thresholded = otsuOutput(crop);
            //cout<<"Area"<<crop.area()<<endl;
//            cvtColor(box,gray,CV_BGR2GRAY);
//            threshold(box,thresholded,0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
//            //imshow("1",gray);
            //imshow("2",thresholded);
//            dilate(thresholded,thresholded,getStructuringElement(MORPH_RECT,Size(7,7)));
//            //imshow("3",thresholded);
//            erode(thresholded,thresholded,getStructuringElement(MORPH_RECT,Size(3,3)));
//            //imshow("4",thresholded);
//            bitwise_not(thresholded,thresholded);
//              imshow("5",thresholded);
//              waitKey(0);
//            inRange(box,Scalar(0,0,0),Scalar(255,255,190),thresholded);
//            erode(thresholded,thresholded,getStructuringElement(MORPH_RECT,Size(2,2)));
//            dilate(thresholded,thresholded,getStructuringElement(MORPH_RECT,Size(7,7)));
//            bitwise_not(thresholded,thresholded);
//            imshow("1",thresholded);
//            waitKey(0);
            findContours(thresholded,contours,CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
            maxArea = 0.0;
            for(int k=0;k<contours.size();k++)
            {
                if(contourArea(contours[k])>maxArea)
                {
                    maxArea = contourArea(contours[k]);
                }
            }
            //cout<<(int)(maxArea*100.0/(thresholded.rows*thresholded.cols))<<" ";
            information[j][i].marked = 0;
            if(maxArea*100.0/(thresholded.rows*thresholded.cols)>=15)
            {
                cout<<j+1<<" "<<i*5<<endl;
                information[j][i].marked = 1;
            }
            information[j][i].x = offsetx+i*stepX;
            information[j][i].y = offsety+j*stepY;
            information[j][i].w = p2.x - information[j][i].x;
            information[j][i].h = p2.y - information[j][i].y;
        }
        cout<<endl;
    }
    debug namedWindow("Sheet1",CV_WINDOW_NORMAL);
    debug imshow("Sheet1",omrSheet);
    debug waitKey(0);
    cout<<endl<<endl<<endl;
}

void getCorrectedOCR(vp corners, vvp cornerContours, Mat input, Mat &sheet1, Mat &sheet2)
{

    // Input Quadilateral or Image plane coordinates
    Point2f inputQuad[4];
    // Output Quadilateral or World plane coordinates
    Point2f outputQuad[4];
    vector<Rect> rects;
    Mat lambda( 2, 4, CV_32FC1 );
    for(int i=0;i<6;i++)
    {
        rects.push_back(boundingRect(cornerContours[i]));
    }
    lambda = Mat::zeros( input.rows, input.cols, input.type() );

    // The 4 points that select quadilateral on the input , from top-left in clockwise order
    // These four pts are the sides of the rect box used as input

    Rect sheet1Rect, sheet2Rect;
    vector<Point2f> sheet1Corners, sheet2Corners;

    sheet1Corners.push_back(Point2f(corners[0].x-rects[0].width/2, corners[0].y+rects[0].height/2));
    sheet1Corners.push_back(Point2f(corners[1].x+rects[1].width/2, corners[1].y+rects[1].height/2));
    sheet1Corners.push_back(Point2f(corners[5].x+rects[5].width/2, corners[5].y-rects[5].height/2));
    sheet1Corners.push_back(Point2f(corners[4].x-rects[4].width/2, corners[4].y-rects[4].height/2));
    sheet1Rect = boundingRect(sheet1Corners);

    sheet2Corners.push_back(Point2f(corners[4].x-rects[4].width/2, corners[4].y+rects[4].height/2));
    sheet2Corners.push_back(Point2f(corners[5].x+rects[5].width/2, corners[5].y+rects[5].height/2));
    sheet2Corners.push_back(Point2f(corners[3].x+rects[3].width/2, corners[3].y-rects[3].height/2));
    sheet2Corners.push_back(Point2f(corners[2].x-rects[2].width/2, corners[2].y-rects[2].height/2));
    sheet2Rect = boundingRect(sheet2Corners);

    inputQuad[0] = Point2f(corners[0].x-rects[0].width/2, corners[0].y+rects[0].height/2);
    inputQuad[1] = Point2f(corners[1].x+rects[1].width/2, corners[1].y+rects[1].height/2);
    inputQuad[2] = Point2f(corners[5].x+rects[5].width/2, corners[5].y-rects[5].height/2);
    inputQuad[3] = Point2f(corners[4].x-rects[4].width/2, corners[4].y-rects[4].height/2);
    // The 4 points where the mapping is to be done , from top-left in clockwise order

    outputQuad[0] = Point2f( 0,0 );
    outputQuad[1] = Point2f( input.cols-1,0);
    outputQuad[2] = Point2f( input.cols-1,input.rows-1);
    outputQuad[3] = Point2f( 0,input.rows-1  );

    lambda = getPerspectiveTransform( inputQuad, outputQuad );

    warpPerspective(input,sheet1,lambda,sheet1.size());
    resize(sheet1,sheet1,sheet1Rect.size());

    inputQuad[0] = Point2f(corners[4].x-rects[4].width/2, corners[4].y+rects[4].height/2);
    inputQuad[1] = Point2f(corners[5].x+rects[5].width/2, corners[5].y+rects[5].height/2);
    inputQuad[2] = Point2f(corners[3].x+rects[3].width/2, corners[3].y-rects[3].height/2);
    inputQuad[3] = Point2f(corners[2].x-rects[2].width/2, corners[2].y-rects[2].height/2);

    lambda = getPerspectiveTransform(inputQuad, outputQuad);
    warpPerspective(input, sheet2, lambda, sheet2.size());
    resize(sheet2, sheet2, sheet2Rect.size());

}

vp findOMR(vvp marks, Mat input, vvp &cornerContours)
{
    // This function finds the location of the markers in an order and stores it in the vector corners

    // Corners stores the location of the marks
    vp corners(6);
    // 0 - top left, 1 - top right, 2 - bottom left, 3 - bottom right, 4 - middle left, 5 - middle right
    vector<Point> useless;
    for(int i=0;i<6;i++)
        cornerContours.push_back(useless);

    vector<double> distances(6);
    // 0 - top left, 1 - top right, 2 - bottom left, 3 - bottom right

    if(marks.size() == 6)
    {
        vector<Moments> mu(marks.size() );
        for( int i = 0; i < marks.size(); i++ )
            mu[i] = moments( marks[i], false );

        // Get the mark centers:
        vector<Point2f> mc( marks.size() );
        for( int i = 0; i < marks.size(); i++ )
            mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );

        // Setting a maximum distance
        for(int i=0; i<marks.size();i++)
        {
            distances[i] = input.rows * input.cols;
        }

        // Finding distances from corners of the image to localize the points
        // TODO: this has to be changed so that the same point isn't stated in two corners
        // Use two loops. Outer for image corner and inner for each mark
        // Maintain a vector stating whether or not the point under consideration has been
        // stated as a corner already
        vector<bool> usedCorner(6);
        for(int i=0;i<6;i++)
            usedCorner[i]=0;
        Point imageCorners[] = {Point2f(0,0),Point2f(input.cols,0),Point2f(0,input.rows),
                                Point2f(input.cols,input.rows),Point2f(0,input.rows/2),Point2f(input.cols,input.rows/2)};

        for(int i=0;i<6;i++)
        {
            int selectedPos = -1;
            for(int j=0; j<6;j++)
            {
                if(absDistance(imageCorners[i],mc[j])<distances[i] && !usedCorner[j])
                {
                    distances[i]=absDistance(imageCorners[i],mc[j]);
                    selectedPos = j;
                }
            }
            corners[i] = mc[selectedPos];
            cornerContours[i] = marks[selectedPos];
            usedCorner[selectedPos] = 1;
        }
    }
    else if(marks.size()<6)
    {
        cout<<"Initialization failed."<<endl;
        errorOccurred = 1;
    }
    else
    {
        cout<<"Initialization failed."<<endl;
        errorOccurred = 2;
    }

    return corners;
}

vvp getLocationOfMarks(Mat thresholded)
{
    vvp contours;
    vvp marks;

    findContours(thresholded, contours, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
    for(int i = 0; i < contours.size(); i++)
    {
        if(contourArea(contours[i])>=0.0001*thresholded.rows*thresholded.cols
                && contourArea(contours[i])<=0.005*thresholded.rows*thresholded.cols)
        {
            marks.push_back(contours[i]);
        }
    }
    return marks;
}

vvp findMarks(Mat input)
{
    // This function finds the location of the markers of the OMR sheet

    Mat inputHSV, thresholded, temp1, temp2;

    // The values of hue have been set according to the hue range of the color red
    // The values of sat have been set high enough to remove unwanted noise
    // TODO
    // The values of val need to be set according to the intensity levels of the image

    int lowH=5, lowS=0, lowV=0, highH=160, highS=255, highV=255;
    vvp contours;
    cvtColor(input,inputHSV,CV_BGR2HSV);

    medianBlur(inputHSV, inputHSV ,9);
    // Since red circles around in the HSV domain, the ranges for hue 0-low  and high-255 have been used
    // instead of 0<=low<high<=255
    bool found=false;
    for(lowS=80;lowS<=180;lowS+=10)
    {
        for(lowV=80;lowV<=180;lowV+=10)
        {
            inRange(inputHSV, Scalar(0,lowS,lowV),Scalar(lowH, highS, highV), temp1);
            inRange(inputHSV, Scalar(highH,lowS,lowV),Scalar(255, highS, highV), temp2);
            bitwise_or(temp1,temp2,thresholded);
//            namedWindow("Thresholded",CV_WINDOW_NORMAL);
//            imshow("Thresholded",thresholded);
//            waitKey(0);
            //cout<<lowS<<" "<<lowV<<endl;
            contours = getLocationOfMarks(thresholded);
            if(contours.size()==6)
            {
                found=true;
                break;
            }
        }
        if(found==true)
            break;
        contours.clear();
    }
    return contours;
}

bool RecognizeMarks(const char *szFileName,COmrResult &Result)
{
    Mat input, sheet1,sheet2;
    input = imread(szFileName);
    if(input.empty())
    {
       cout<<"Could not read image. Please check if file path is valid."<<endl;
       return 0;
    }
    vvp marks, cornerContours;
    vp corners;
    marks = findMarks(input);
    corners = findOMR(marks,input, cornerContours);
    if(errorOccurred == 1)
    {
        cout<<"The program couldn't find the marks correctly. Please ensure all marks are clearly visible. Terminating."<<endl;
        return 0;
    }
    if(errorOccurred == 2)
    {
        cout<<"The program couldn't find the marks correctly. Please ensure there are no red objects in the image. Terminating."<<endl;
        return 0;
    }
    getCorrectedOCR(corners,cornerContours,input,sheet1,sheet2);
    extractInformation(sheet1,Result.Black);
    extractInformation(sheet2,Result.White);
}

int main()
{
    const char name[] = "/home/codestation/Documents/OpenCV/OMR-Computer-Vision/Test/Test/2/1.jpg";
    COmrResult Result;
    RecognizeMarks(name, Result);
}
