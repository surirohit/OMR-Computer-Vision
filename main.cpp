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

void extractInformation(Mat omrSheet)
{
    Mat gray,thresholded, box;
    double offsetx = 0.0408*omrSheet.cols;
    double offsety = 0.0488*omrSheet.rows;
    double stepX = (omrSheet.cols-offsetx+0.0139*omrSheet.cols)/21.0;
    double stepY = (omrSheet.rows-offsety)/25.0;
    for(int j=0;j<25;j++)
    {
        for(int i=0; i<21; i++)
        {
            debug rectangle(omrSheet,Rect(Point(offsetx+i*stepX,offsety+j*stepY),
                                    Point(offsetx+(i+1)*stepX,offsety+(j+1)*stepY)),Scalar(0,255,0));
            Point p2 = Point(offsetx+(i+1)*stepX,offsety+(j+1)*stepY);
            if(p2.x>=omrSheet.cols)
            {
                p2.x=omrSheet.cols - 1;
            }
            if(p2.y>=omrSheet.rows)
            {
                p2.y = omrSheet.rows-1;
            }
            Rect crop = Rect(Point(offsetx+i*stepX,offsety+j*stepY),p2);
            box = omrSheet(crop);
            cvtColor(box,gray,CV_BGR2GRAY);
            threshold(gray,thresholded,0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
            dilate(thresholded,thresholded,getStructuringElement(MORPH_RECT,Size(3,3)));
            //cout<<countNonZero(thresholded)*100.0/(thresholded.rows*thresholded.cols)<<endl;
            if(countNonZero(thresholded)*100.0/(thresholded.rows*thresholded.cols)<70.0)
            {
                cout<<j+1<<" "<<i*5<<endl;
            }
        }
    }
    debug namedWindow("Sheet1",CV_WINDOW_NORMAL);
    debug imshow("Sheet1",omrSheet);
    debug waitKey(0);
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

double absDistance(Point2f point1, Point2f point2)
{
    return sqrt(pow((point1.x-point2.x),2)+pow((point1.y-point2.y),2));
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

    // Removing contours which are very big or very small
    if(contours.size()>6)
    {
        for(int i = 0; i < contours.size(); i++)
        {
            if(contourArea(contours[i])>=0.0001*thresholded.rows*thresholded.cols
                    && contourArea(contours[i])<=0.001*thresholded.rows*thresholded.cols)
            {
                marks.push_back(contours[i]);
            }
        }
    }
    else
    {
        marks = contours;
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

    int lowH=9, lowS=80, lowV=80, highH=160, highS=255, highV=255;
    vvp contours;
    cvtColor(input,inputHSV,CV_BGR2HSV);

    medianBlur(inputHSV, inputHSV ,5);
    // Since red circles around in the HSV domain, the ranges for hue 0-low  and high-255 have been used
    // instead of 0<=low<high<=255

    inRange(inputHSV, Scalar(0,lowS,lowV),Scalar(lowH, highS, highV), temp1);
    inRange(inputHSV, Scalar(highH,lowS,lowV),Scalar(255, highS, highV), temp2);
    bitwise_or(temp1,temp2,thresholded);

    contours = getLocationOfMarks(thresholded);
    return contours;
}

int main()
{
    Mat input, sheet1,sheet2;
    input = imread("/home/codestation/Documents/OpenCV/OMR-Computer-Vision/Test/1.jpg");
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
    extractInformation(sheet1);
}
