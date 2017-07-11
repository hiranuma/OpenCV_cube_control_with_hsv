#pragma warning(disable : 4819)
#include <opencv2/opencv.hpp>
#pragma warning(default: 4819)
#include <opencv2/imgcodecs.hpp>
#ifdef _DEBUG
#pragma comment(lib, "Debug/opencv_world320d.lib")
#else
#pragma comment(lib, "Release/opencv_world320.lib")
#endif
#include "GL/freeglut.h"

using namespace cv;
using namespace std;

string win_gl = "win_gl";
string win_cv_src = "win_cv_src";
string win_cv_bin = "win_cv_bin";

int win_w = 480, win_h = 480;
int zoom = 20;
double rot_x, rot_y;

void drawCallback(void* userdata)
{
	glClear(GL_COLOR_BUFFER_BIT);
	glLoadIdentity();

	gluLookAt(0.0, 0.0, zoom, // view pos
		0.0, 0.0, 0.0, // view dist
		0.0, 1.0, 0.0); // view vector

	// Rotate model
	glRotated(rot_x / 3, 0.0, 1.0, 0.0);
	glRotated(rot_y / 3, 1.0, 0.0, 0.0);

	GLfloat col[] = { 1.0, 1.0, 0.0, 1.0 };
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, col);
	glutSolidCube(1.0);
}

void init()
{
	GLfloat pos[] = { 100.0, 100.0, 100.0, 1.0 };
	GLfloat col_d[] = { 1.0, 1.0, 1.0, 1.0 };
	GLfloat col_a[] = { 0.1, 0.1, 0.1, 1.0 };
	glLightfv(GL_LIGHT0, GL_POSITION, pos);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, col_d);
	glLightfv(GL_LIGHT0, GL_AMBIENT, col_a);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_CULL_FACE);
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(10.0, (GLdouble)win_w / win_h, 0.1, 1000);
	glMatrixMode(GL_MODELVIEW);
}

void mouseCallback(int event, int x, int y, int flags, void* userdata)
{
	cout << event << " " << x << " " << y << " " << flags << endl;
}

int main(int argc, char* argv[])
{
	Mat img_src;
	Mat img_hsv;
	vector<Mat> vec_hsv;
	VideoCapture cap(0);

	if (!cap.isOpened()) {
		cout << "open failed" << endl;
		return -1;
	}
	if (!cap.open(0)) {
		cout << "camera not failed" << endl;
		return -1;
	}

	namedWindow(win_gl, WINDOW_OPENGL);
	resizeWindow(win_gl, win_w, win_h);
	createTrackbar("zoom", win_gl, &zoom, 100);

	setOpenGlContext(win_gl);
	setMouseCallback(win_gl, mouseCallback);
	setOpenGlDrawCallback(win_gl, drawCallback);

	glutInit(&argc, argv);
	init();

	while (1) {
		cap >> img_src;

		cvtColor(img_src, img_hsv, COLOR_BGR2HSV);
		split(img_hsv, vec_hsv);
		threshold(vec_hsv[0], vec_hsv[0], 30, 255, THRESH_TOZERO_INV); // Maximum
		threshold(vec_hsv[0], vec_hsv[0], 0, 255, THRESH_BINARY); // Minimum

		// Remove noise
		Mat e4 = (Mat_<uchar>(3, 3) << 0, 1, 0, 1, 1, 1, 0, 1, 0); // 4 neighborhood
		morphologyEx(vec_hsv[0], vec_hsv[0], MORPH_CLOSE, e4, Point(-1, -1), 5);

		// Labeling
		Mat img_lab;
		Mat stats, center;
		int nLabels = connectedComponentsWithStats(vec_hsv[0], img_lab, stats, center);
		// Pick up the largest area label
		vector <int>area;
		for (int i = 1; i < nLabels; i++)
			area.push_back(stats.at<int>(i, CC_STAT_AREA));
		vector<int>::iterator it = max_element(area.begin(), area.end());
		size_t index = distance(area.begin(), it) + 1;
		compare(img_lab, index, vec_hsv[0], CMP_EQ);

		// Calculate gap from center
		Point cog(center.at<double>(index, 0),
			center.at<double>(index, 1));
		rot_x = cog.x - vec_hsv[0].cols / 2;
		rot_y = cog.y - vec_hsv[0].rows / 2;

		// Draw
		circle(img_src, cog, 10, Scalar(0, 0, 255), -1);
		imshow(win_cv_src, img_src);
		imshow(win_cv_bin, vec_hsv[0]);
		updateWindow(win_gl);
		if ((char)waitKey(1) >= 0) break;
	}

	return 0;
}