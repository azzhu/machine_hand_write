// machine_hand_write.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
#include "D:\myfile\zqjtools.hpp"

using namespace std;
using namespace cv;
using namespace az;

//���ֽṹ������
struct CCPoint
{
	int ind;					//���
	int out_dgr;				//����
	cv::Point p;				//����ֵ
	std::vector<cv::Point> p_nb;	//���ڵĵ�
};
//��ʼ���������ݽṹ
struct BEPoint
{
	Point BeginPoint;
	Point EndPoint;
};

Mat getFrCam();
cv::Mat thinImage(cv::Mat & src, const int maxIterations = -1);
void drawPs(Mat* pic, std::vector<cv::Point> ps);
void zhengli_ps(Mat *pic, std::vector<cv::Point> &ps);
CCPoint* Point2CCPoint(cv::Point ps, std::vector<CCPoint> &cps);
void merge_ps(std::vector<cv::Point> ps1, std::vector<cv::Point>ps2, std::vector<cv::Point>& res, int h, int w);
void harris_dtct(Mat pic, vector<cv::Point>& corners);
void oneout_dtct(Mat src, vector<cv::Point> &ps);
void expand_p(Mat* pic, vector<cv::Point> &ps);
void neighbor_ps(Mat* pic, Point p, std::vector<Point> ps, std::vector<cv::Point> &ps_nb);
void ps2cps(Mat* pic, std::vector<cv::Point> ps, std::vector<CCPoint> &cps);
void classifyByOutdgr(vector<CCPoint> cps, vector<CCPoint> &cps_o1, vector<CCPoint> &cps_o2, vector<CCPoint> &cps_o3);
int dist_2p(int x, int y, CCPoint cp);
void del_cp(vector<CCPoint> &cps, CCPoint cp);
void del_p(vector<Point> &ps, Point p);
void next_nbcp(CCPoint cp, Point &cp_nt);
bool next_nbcp(CCPoint cp1, CCPoint cp2, Point &cp_nt);
void stroke_plan(Mat* pic, vector<CCPoint> cps, vector<vector<Point>> &spp);
void insideSetTo0(Mat *pic);
void multiout_dtct();
void write(vector<vector<Point>> spp, cv::Size s, std::vector<cv::Point> ps);
void jisuanguaidian(vector<pair<Point, int>> pada, int par_kd, int par_jd, vector<Point> &ps);
void getAllFeaturePoints(Mat *img, int par_bc, int par_kd, int par_jd, vector<Point> &rps);
void merge_ps_2(Mat* img, vector<Point> ps1, vector<Point>ps2, vector<Point> &res);

//param
bool harrisOrShiTomasi = 1;		//0:ShiTomasi,1:harris
bool merOneoutD = 1;
float addRd = 1.0;
int kerDist = 15;
bool cirOrPoi = 1;
bool thin = 1;
int writeThickness = 6;
int writeShakeRange = 0;
int spd1 = 10;
int spd2 = 20;
int pic_w, pic_h;
bool saveIt = 0;
bool drawFeatureImg = 1;
string fn = "C:\\Users\\Administrator\\Desktop\\QQ��ͼ20161227165646.png";

//use����o��������ȡͼ��ʼд�֣���r������дһ��

#define DEBUG_MODE

//vector<pair<Point, int>> quanju_pada;
void testcode()
{
	//string picfn0 = "D:\\csimg\\�Ż������̫���������㡿\\c--.jpg"; 
	//Mat zai0 = imread(picfn0, 0);
	////dilate(zai0, zai0, mk);
	////imwrite("D:\\csimg\\�Ż������̫���������㡿\\fushi.bmp", zai0);

	////xihua
	//cv::threshold(zai0, zai0, 128, 1, cv::THRESH_BINARY_INV);
	//Mat zai = thinImage(zai0);
	//zai *= 255;
	//imwrite("D:\\csimg\\�Ż������̫���������㡿\\ttt.bmp", zai);

	string picfn0 = "D:\\csimg\\�Ż������̫���������㡿\\tian.jpg";
	Mat zai = imread(picfn0, 0);

	std::vector<cv::Point> ps7;

	int kkk = 1
		;

	if (kkk == 1)
		getAllFeaturePoints(&zai, 30, 5, 10, ps7);	//30��5��10
	else
	{
		harris_dtct(zai, ps7);
		zhengli_ps(&zai, ps7);
		//oneout_dtct(zai, ps7);
	}

	Mat nji;
	cvtColor(zai, nji, CV_GRAY2BGR);
	for (int i = 0; i < ps7.size(); i++)
	{
		nji.at<Vec3b>(ps7[i]) = Vec3b(0, 0, 255);
		//circle(nji, ps7[i], 3, Scalar(0, 0, 255), 2);
	}
	imwrite("D:\\csimg\\�Ż������̫���������㡿\\temp.bmp", nji);

	return;	
}

string srcfn, fmimgfn, txtfn;
void ������()
{//test code:��ȡһ���ļ����ڵ�ͼ�����ɴ��������ͼ��txt������
	vector<string> fns;
	az::getAllFilesPath("D:\\csimg\\������\\sample", fns);

	for (int i = 0; i < fns.size(); i++)
	{
		srcfn = fns[i];
		fmimgfn = srcfn + "fm.bmp";
		txtfn = srcfn + ".txt";
		Mat zai = imread(srcfn, 0);
		//��������
		std::vector<cv::Point> ps;
		getAllFeaturePoints(&zai, 30, 5, 10, ps);
		//harris_dtct(zai, ps);
		//zhengli_ps(&zai, ps);
		drawPs(&zai, ps);
		std::vector<CCPoint> cps;
		ps2cps(&zai, ps, cps);
		ofstream of(txtfn);
		for (int i = 0; i < cps.size(); i++)
		{
			int x, y, flg;
			x = cps[i].p.x + 1;
			y = cps[i].p.y + 1;
			if (cps[i].p_nb.size() == 1)
				flg = 1;
			else
				flg = 0;
			of << x << " " << y << " " << flg << endl;
		}
		of.close();
		waitKey(33);
	}

	system("pause");
	return ;
}

int _tmain(int argc, char* argv[])
{
	testcode();
	//������();
	return 0;

	//COMMAND_LINE_MODE
#ifndef DEBUG_MODE
	//load param
	if (argc != 2)
	{
		cout << "�����в������������ļ�û�м��أ�" << endl;
		return 0;
	}
	char* config = argv[1];
	char filename[MAX_PATH];
	char addRdst[MAX_PATH];
	GetPrivateProfileString("src", "imgPath", "null", filename, MAX_PATH, config);
	harrisOrShiTomasi = GetPrivateProfileInt("param", "harrisOrShiTomasi", 0, config);
	merOneoutD = GetPrivateProfileInt("param", "merOneoutD", 1, config);
	GetPrivateProfileString("param", "addRd", "1.2", addRdst, MAX_PATH, config);
	addRd = (float)_tstof(addRdst);
	kerDist = GetPrivateProfileInt("param", "kerDist", 10, config);
	//cirOrPoi = GetPrivateProfileInt("param", "cirOrPoi", 0, config);
	//thin = GetPrivateProfileInt("param", "thin", 1, config);
	writeThickness = GetPrivateProfileInt("param", "writeThickness", 6, config);
	writeShakeRange = GetPrivateProfileInt("param", "writeShakeRange", 0, config);
	spd1 = GetPrivateProfileInt("param", "spd1", 10, config);
	spd2 = GetPrivateProfileInt("param", "spd2", 20, config);
	saveIt = 0;
	drawFeatureImg = GetPrivateProfileInt("param", "drawFeatureImg", 1, config);
	//��ͼ��
	Mat picsrc = imread(filename);
	imshow("ԭͼ", picsrc);
	cvtColor(picsrc, picsrc, CV_BGR2GRAY);
	//Mat pic = getFrCam();
	cv::threshold(picsrc, picsrc, 128, 1, cv::THRESH_BINARY_INV);
	Mat pic = thinImage(picsrc);
	pic *= 255;
	pic_w = pic.cols;
	pic_h = pic.rows;
#endif

#ifdef DEBUG_MODE
	//��ͼ��
	Mat picsrc = imread(fn);
	imshow("ԭͼ", picsrc);
	cvtColor(picsrc, picsrc, CV_BGR2GRAY);
	//Mat pic = getFrCam();
	cv::threshold(picsrc, picsrc, 128, 1, cv::THRESH_BINARY_INV);
	Mat pic = thinImage(picsrc);
	pic *= 255;
	pic_w = pic.cols;
	pic_h = pic.rows;
#endif

	long st = getTime();

	//��������
	std::vector<cv::Point> ps;
	harris_dtct(pic, ps);
	zhengli_ps(&pic, ps);
	if (drawFeatureImg)
		drawPs(&pic, ps);

	waitKey();

	//ת��ΪCCPoint
	std::vector<CCPoint> cps;
	ps2cps(&pic, ps, cps);
	
	//�ʻ��滮
	vector<vector<Point>> spp;
	stroke_plan(&pic, cps, spp);

	long ed = getTime();
	cout << ed - st << "ms" << endl;

	//ģ��д��
	cv::Size sss = pic.size();
	write(spp, sss, ps);

	cv::waitKey();
	return 0;
}

void picInverse(Mat* pic)
{//��ֵͼ��ɫ
	if (pic->channels() != 1)
		return;
	int c = pic->cols;
	int r = pic->rows;
	for (int i = 0; i < r; i++)
	{
		uchar* data = pic->ptr<uchar>(i);
		for (int j = 0; j < c; j++)
		{
			*data++ = 255 - *data;
		}
	}
}

Mat rotateImage(Mat* pic, int angle, bool clockwise)
{
	IplImage* src = &IplImage(*pic);
	angle = abs(angle) % 180;
	if (angle > 90)
	{
		angle = 90 - (angle % 90);
	}
	IplImage* dst = NULL;
	int width =
		(double)(src->height * sin(angle * CV_PI / 180.0)) +
		(double)(src->width * cos(angle * CV_PI / 180.0)) + 1;
	int height =
		(double)(src->height * cos(angle * CV_PI / 180.0)) +
		(double)(src->width * sin(angle * CV_PI / 180.0)) + 1;
	int tempLength = sqrt((double)src->width * src->width + src->height * src->height) + 10;
	int tempX = (tempLength + 1) / 2 - src->width / 2;
	int tempY = (tempLength + 1) / 2 - src->height / 2;
	int flag = -1;

	dst = cvCreateImage(cvSize(width, height), src->depth, src->nChannels);
	cvZero(dst);
	IplImage* temp = cvCreateImage(cvSize(tempLength, tempLength), src->depth, src->nChannels);
	cvZero(temp);

	cvSetImageROI(temp, cvRect(tempX, tempY, src->width, src->height));
	cvCopy(src, temp, NULL);
	cvResetImageROI(temp);

	// clockwise Ϊtrue��˳ʱ����ת������Ϊ��ʱ����ת
	if (clockwise)
		flag = 1;

	float m[6];
	int w = temp->width;
	int h = temp->height;
	m[0] = (float)cos(flag * angle * CV_PI / 180.);
	m[1] = (float)sin(flag * angle * CV_PI / 180.);
	m[3] = -m[1];
	m[4] = m[0];
	// ����ת��������ͼ���м�  
	m[2] = w * 0.5f;
	m[5] = h * 0.5f;
	//  
	CvMat M = cvMat(2, 3, CV_32F, m);
	cvGetQuadrangleSubPix(temp, dst, &M);
	cvReleaseImage(&temp);
	Mat dstm = cvarrToMat(&dst);
	return dstm;
}

Mat getFrCam()
{
	cv::Mat frame;
	cv::Mat frame2;
	cv::VideoCapture pCapture(0);

	while (1)
	{
		if (!pCapture.read(frame))
			break;

		frame.copyTo(frame2);
		Rect rt(220, 140, 200, 200);
		rectangle(frame2, rt, Scalar(255, 255, 0), 2);
		cv::imshow("frame", frame2);

		char c = cvWaitKey(33);
		if (c == 'o')
		{
			cvtColor(frame, frame, CV_BGR2GRAY);
			Mat fmroi = frame(rt);
			if (thin)
			{
				cv::threshold(fmroi, fmroi, 128, 1, cv::THRESH_BINARY_INV);
				Mat fmthin = thinImage(fmroi);
				fmthin *= 255;
				return fmthin;
			}
			else
			{
				picInverse(&fmroi);		//��ɫ
				return fmroi;
			}
		}
	}
}

cv::Mat thinImage(cv::Mat & src, const int maxIterations)
{
	assert(src.type() == CV_8UC1);
	cv::Mat dst;
	int width = src.cols;
	int height = src.rows;
	src.copyTo(dst);
	int count = 0;  //��¼��������  
	while (true)
	{
		count++;
		if (maxIterations != -1 && count > maxIterations) //���ƴ������ҵ�����������  
			break;
		std::vector<uchar *> mFlag; //���ڱ����Ҫɾ���ĵ�  
		//�Ե���  
		for (int i = 0; i < height; ++i)
		{
			uchar * p = dst.ptr<uchar>(i);
			for (int j = 0; j < width; ++j)
			{
				//��������ĸ����������б��  
				//  p9 p2 p3  
				//  p8 p1 p4  
				//  p7 p6 p5  
				uchar p1 = p[j];
				if (p1 != 1) continue;
				uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
				uchar p8 = (j == 0) ? 0 : *(p + j - 1);
				uchar p2 = (i == 0) ? 0 : *(p - dst.step + j);
				uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - dst.step + j + 1);
				uchar p9 = (i == 0 || j == 0) ? 0 : *(p - dst.step + j - 1);
				uchar p6 = (i == height - 1) ? 0 : *(p + dst.step + j);
				uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + dst.step + j + 1);
				uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + dst.step + j - 1);
				if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6)
				{
					int ap = 0;
					if (p2 == 0 && p3 == 1) ++ap;
					if (p3 == 0 && p4 == 1) ++ap;
					if (p4 == 0 && p5 == 1) ++ap;
					if (p5 == 0 && p6 == 1) ++ap;
					if (p6 == 0 && p7 == 1) ++ap;
					if (p7 == 0 && p8 == 1) ++ap;
					if (p8 == 0 && p9 == 1) ++ap;
					if (p9 == 0 && p2 == 1) ++ap;

					if (ap == 1 && p2 * p4 * p6 == 0 && p4 * p6 * p8 == 0)
					{
						//���  
						mFlag.push_back(p + j);
					}
				}
			}
		}

		//����ǵĵ�ɾ��  
		for (std::vector<uchar *>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
		{
			**i = 0;
		}

		//ֱ��û�е����㣬�㷨����  
		if (mFlag.empty())
		{
			break;
		}
		else
		{
			mFlag.clear();//��mFlag���  
		}

		//�Ե���  
		for (int i = 0; i < height; ++i)
		{
			uchar * p = dst.ptr<uchar>(i);
			for (int j = 0; j < width; ++j)
			{
				//��������ĸ����������б��  
				//  p9 p2 p3  
				//  p8 p1 p4  
				//  p7 p6 p5  
				uchar p1 = p[j];
				if (p1 != 1) continue;
				uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
				uchar p8 = (j == 0) ? 0 : *(p + j - 1);
				uchar p2 = (i == 0) ? 0 : *(p - dst.step + j);
				uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - dst.step + j + 1);
				uchar p9 = (i == 0 || j == 0) ? 0 : *(p - dst.step + j - 1);
				uchar p6 = (i == height - 1) ? 0 : *(p + dst.step + j);
				uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + dst.step + j + 1);
				uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + dst.step + j - 1);

				if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6)
				{
					int ap = 0;
					if (p2 == 0 && p3 == 1) ++ap;
					if (p3 == 0 && p4 == 1) ++ap;
					if (p4 == 0 && p5 == 1) ++ap;
					if (p5 == 0 && p6 == 1) ++ap;
					if (p6 == 0 && p7 == 1) ++ap;
					if (p7 == 0 && p8 == 1) ++ap;
					if (p8 == 0 && p9 == 1) ++ap;
					if (p9 == 0 && p2 == 1) ++ap;

					if (ap == 1 && p2 * p4 * p8 == 0 && p2 * p6 * p8 == 0)
					{
						//���  
						mFlag.push_back(p + j);
					}
				}
			}
		}

		//����ǵĵ�ɾ��  
		for (std::vector<uchar *>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
		{
			**i = 0;
		}

		//ֱ��û�е����㣬�㷨����  
		if (mFlag.empty())
		{
			break;
		}
		else
		{
			mFlag.clear();//��mFlag���  
		}
	}
	return dst;
}

void drawPs(Mat* pic, std::vector<cv::Point> ps)
{
	cout << ps.size() << endl;
	Mat img;
	pic->copyTo(img);
	//img.setTo(0);
	cv::cvtColor(img, img, CV_GRAY2BGR);
	std::vector<cv::Point>::iterator it = ps.begin(), ite = ps.end();
	for (; it != ite; it++)
	{
		if (cirOrPoi)
		{
			cv::circle(img, *it, 3, Scalar(0, 0, 255), 2);
		}
		else
		{
			img.at<cv::Vec3b>(*it) = Vec3b(0, 0, 255);
			//cv::rectangle(img, Rect(it->x - 5, it->y - 5, 10, 10), Scalar(0, 0, 255), 1);
			cv::putText(img, to_string(it->x), *it, CV_FONT_HERSHEY_SIMPLEX, 
				0.4, Scalar(255, 255, 255));
		}
	}
	cv::imwrite(fmimgfn, img);
	//cv::namedWindow("FeatureImg");
	//cvMoveWindow("FeatureImg", 1280 + 300, 350);
	//cv::imshow("FeatureImg", img);
}

//�ѿ������ⲿ�ĵ�������ԭ���Ƶ��ڲ�
void zhengli_ps(Mat *pic, std::vector<cv::Point> &ps)
{
	std::vector<cv::Point> res_ps;
	int r = pic->rows, c = pic->cols;
	std::vector<cv::Point>::iterator itb = ps.begin();
	std::vector<cv::Point>::iterator ite = ps.end();
	for (; itb != ite; itb++)
	{
		int x = itb->x, y = itb->y;
		for (int kern = 0; kern <= 15; kern++)	//��Զ�������15������
		{
			for (int m = x - kern; m <= x + kern; m++)
			{
				for (int n = y - kern; n <= y + kern; n++)
				{
					if (m == x - kern ||
						m == x + kern ||
						n == y - kern ||
						n == y + kern)		//��������
						if (m >= 0 &&
							n >= 0 &&
							m < c&&
							n < r)			//û��Խ��
							if (pic->at<uchar>(n, m)>200)
							{
								res_ps.push_back(cv::Point(m, n));
								goto mark49;
							}
				}
			}
		}
	mark49:;
	}

	ps.clear();
	ps = res_ps;
}

//��CCPoint������������Point��Ӧ��CCPoint�㣬����ָ��
CCPoint* Point2CCPoint(cv::Point ps, std::vector<CCPoint> &cps)
{
	std::vector<CCPoint>::iterator it = cps.begin(), ite = cps.end();
	for (; it != ite; it++)
	{
		if (ps == it->p)
			return &(*it);
	}
	cout << "û���ҵ���֮��Ӧ��CCPoint�㣺CCPoint Point2CCPoint()" << endl;
	return nullptr;
}

//����ϲ��㷨�����ˣ������̫���Ļ����ͻᱻ�ϲ���һ����
void merge_ps(std::vector<cv::Point> ps1, std::vector<cv::Point>ps2, std::vector<cv::Point>& res, int h, int w)
{
	//param
	int dist = 2;

	Mat pic(h, w, CV_8UC1);
	pic.setTo(0);
	std::vector<std::vector<cv::Point>> ps;

	std::vector<cv::Point>::iterator it = ps1.begin();
	for (; it != ps1.end(); it++)
	{
		circle(pic, *it, dist, 255, dist);
	}
	std::vector<cv::Point>::iterator it2 = ps2.begin();
	for (; it2 != ps2.end(); it2++)
	{
		circle(pic, *it2, dist, 255, dist);
	}
	imwrite("D:\\csimg\\�Ż������̫���������㡿/temphebing.jpg", pic);

	az::findLTQY(&pic, ps);
	az::regions2points(ps, res);
}

//�µ�����ϲ��㷨���ѵ����ȼ�⼯�Ϸ�ǰ�棬���ȱ��������ȼ��Ľ��
void merge_ps_2(Mat* img, vector<Point> ps1, vector<Point>ps2, vector<Point> &res)
{
	//���㷨�����������б���ÿ������ԣ�Ȼ���ж�ÿ������Ƿ�ֲ�������
	//���������ϲ�,������ϲ�
	bool can_merge(Mat smallimg, Point p1, Point p2);
	bool hasthispoint(Point p, vector<Point> ps);

	int dist = 15;	//�������С�ڶ���ʱ��ʼ�ж�
	int pad = 10;	//�������������������أ�Ȼ������smallimg
	vector<Point> res_;	//������Ҫɾ���ĵ�
	int w = img->cols, h = img->rows;

	//����ı�����Ѱ��Ҫɾ���ĵ�
	vector<Point>::iterator itb1 = ps1.begin(), ite1 = ps1.end(),
							itb2 = ps2.begin(), ite2 = ps2.end();
	for (; itb1 != ite1; itb1++)
	{
		Point p1 = *itb1;
		for (; itb2 != ite2; itb2++)
		{
			Point p2 = *itb2;
			int dist_ = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
			if (dist_ <= dist)
			{
				int x_min = min(p1.x - pad, p2.x - pad),
					x_max = max(p1.x + pad, p2.x + pad), 
					y_min = min(p1.y - pad, p2.y - pad),
					y_max = max(p1.y + pad, p2.y + pad);
				if (x_min < 0) x_min ^= x_min;
				if (y_min < 0) y_min ^= y_min;
				if (x_max > w) x_max = w;
				if (y_max > h) y_max = h;
				Mat smallimg = (*img)(Rect(x_min, y_min, x_max - x_min, y_max - y_min));
				if (can_merge(smallimg,
					Point(p1.x - x_min, p1.y - y_min),
					Point(p2.x - x_min, p2.y - y_min)))
					res_.push_back(p2);	//ɾp2������p1;
			}
		}
	}

	//����ı������ų�Ҫɾ���ĵ㲢�ϲ�
	res = ps1;
	itb2 = ps2.begin();
	for (; itb2 != ite2; itb2++)
	{
		if (!hasthispoint(*itb2, res_))
			res.push_back(*itb2);
	}
}
//�ж�һ����������û�������
bool hasthispoint(Point p, vector<Point> ps)
{
	if (ps.size() != 0)
	{
		vector<Point>::iterator it = ps.begin(), ite = ps.end();
		for (; it != ite; it++)
		{
			if (p == *it)
				return true;
		}
	}
	return false;
}
//�ж������ܷ�ϲ�
bool can_merge(Mat smallimg, Point p1, Point p2)
{
	//smallimg:�Ǹ����㸽���ֲ�С�����ͼ��

	//�����������ȣ�ֱ�Ӻϲ�
	if (p1 == p2)	return true;

	Mat tempimg;
	smallimg.copyTo(tempimg);
	//���p1���������е�
	vector<Point> ps;
	az::pot_erg(&tempimg, p1, ps);
	//����Ѱ�������Ƿ���p2
	vector<Point>::iterator itb = ps.begin(), ite = ps.end();
	for (; itb != ite; itb++)
	{
		//�ҵ��ˣ��������������Ժϲ�
		if (*itb == p2)
			return true;
	}
	//û���ҵ��������������ܺϲ�
	return false;
}

void harris_dtct(Mat pic, vector<cv::Point>& corners)
{
	cv::threshold(pic, pic, 100, 255, cv::ThresholdTypes::THRESH_BINARY);
	cv::Mat pic2, pic3;
	pic.copyTo(pic2);
	pic.copyTo(pic3);

	//Harris�ǵ���
	cv::Mat harriscn;
	cv::cornerHarris(pic, harriscn, 7, 5, 0.1);
	cv::threshold(harriscn, harriscn, 0.0001, 255, cv::THRESH_BINARY);
	//���Ͻ��
	cv::cvtColor(pic, pic, CV_GRAY2BGR);
	cv::Mat mask;
	harriscn.convertTo(mask, CV_8UC1);
	cv::medianBlur(mask, mask, 5);
	cv::medianBlur(mask, mask, 3);
	cv::medianBlur(mask, mask, 3);
	//cv::imshow("mask", mask);
	std::vector<std::vector<cv::Point>> vss;
	az::findLTQY(&mask, vss);
	std::vector<cv::Point> pss;
	az::regions2points(vss, pss);

	if (harrisOrShiTomasi)
	{
		//harris�ǵ���
		//�ϲ������ȼ����
		if (merOneoutD)
		{
			std::vector<cv::Point> ps_one, res;
			oneout_dtct(pic3, ps_one);
			//merge_ps(pss, ps_one, res, pic3.rows, pic3.cols);
			merge_ps_2(&pic, ps_one, pss, res);
			pss = res;
		}
		corners = pss;
	}
	else
	{
		//Shi-Tomasi�ǵ���
		int num = pss.size()*addRd;
		cv::goodFeaturesToTrack(pic2, corners, num, 0.01, 20);
		//�ϲ������ȼ����
		if (merOneoutD)
		{
			std::vector<cv::Point> ps_one, res;
			oneout_dtct(pic3, ps_one);
			merge_ps(corners, ps_one, res, pic3.rows, pic3.cols);
			corners = res;
		}
	}
}

void oneout_dtct(Mat src, vector<cv::Point> &ps)
{
	//��������Ч�����Ǻܺõģ�û�������©��
	//����һ�����ص�ʱ�����жϣ�����Ϊ��һ�������ȵ�

	//����
	int kern = 3;

	int flg = 0;	//���ڼ�¼һ���������ܰ˸�ֵ��Ϊ��ĸ���
	cv::Mat pic;
	cv::threshold(src, pic, 100, 255, cv::ThresholdTypes::THRESH_BINARY);	//��ֵ��
	//cv::imshow("erzhihua", pic);
	int r = pic.rows, c = pic.cols;
	for (int i = 0; i < r; i++)
	{
		uchar* data = pic.ptr<uchar>(i);
		for (int j = 0; j < c; j++)
		{
			//�����ںڵװ��������
			if (*data++ > 200 &&
				i >= kern / 2 && i <= (r - kern / 2 - 1) &&
				j >= kern / 2 && j <= (c - kern / 2 - 1))			//�߽�ͺ�ɫ���������ж�
			{
				flg = 0;		//����
				vector<Point> pstemp;
				pstemp.clear();
				for (int i_ = i - 1; i_ <= i + 1; i_++)
					for (int j_ = j - 1; j_ <= j + 1; j_++)
					{
						if ((i_ == i) && (j_ == j))
							;
						else
						{
							cv::Point p(j_, i_);
							if (pic.at<uchar>(p)>200)
							{
								pstemp.push_back(p);
								flg++;
							}
						}
					}
				if (flg == 1)
					ps.push_back(Point(j, i));
				else if (flg == 2)	//�ж��Ƿ�Ϊһ������
				{
					int x1 = pstemp[0].x, y1 = pstemp[0].y;
					int x2 = pstemp[1].x, y2 = pstemp[1].y;
					if ((x1 == x2&&y1 == y2 - 1) || 
						(x1 == x2&&y1 == y2 + 1) ||
						(y1 == y2&&x1 == x2 - 1) ||
						(y1 == y2&&x1 == x2 + 1))
						ps.push_back(Point(j, i));
				}
			}
		}
	}
}

void expand_p(Mat* pic, vector<cv::Point> &ps)
{
	int kern = 4;	//����4������
	int h = pic->rows, w = pic->cols;
	vector<cv::Point>::iterator itb = ps.begin(), ite = ps.end();
	for (; itb != ite; itb++)
	{
		int x = itb->x;
		int y = itb->y;
		pic->at<uchar>(y, x) = 127;
		for (int i = x - kern; i <= x + kern; i++)
			for (int j = y - kern; j <= y + kern; j++)
				if (i >= 0 && i < w&&
					j >= 0 && j < h)		//�Ƿ����
					if (pic->at<uchar>(j, i)>200)		//�׵�
						pic->at<uchar>(j, i) = 127;		//�û�
	}
	return;
}

//������ͨ����Ѱ�����ڵĵ㣺ԭʼͼ����ʼ�㡢���������㼯��dst���ڵĵ㼯
void neighbor_ps(Mat* pic, Point p, std::vector<Point> ps, std::vector<cv::Point> &ps_nb)
{
	if (pic->channels() != 1)
	{
		cout << "Ӧ���뵥ͨ���Ҷ�ͼ��void neighbor_ps()" << endl;
		return;
	}

	ps_nb.clear();
	Mat src;
	pic->copyTo(src);
	std::vector<cv::Point> bdps;	//��������Ѱ��ʱ�߽�㼯
	std::vector<cv::Point> bdps_;	//����Ѱ��һ�ֱ������ʱ�߽�㼯

	bdps.push_back(p);		//p��Ϊ���
	src.at<uchar>(p) = 0;	//����

	while (1)
	{
		std::vector<cv::Point>::iterator 
			//it = bdps.begin(), 
			it2 = bdps.begin(), 
			ite = bdps.end();
		//Ѱ����һ�ֱ߽�
		bdps_.clear();
		for (; it2 != ite; it2++)
		{
			for (int i = it2->x - 1; i <= it2->x + 1; i++)
				for (int j = it2->y - 1; j <= it2->y + 1; j++)
					if (i >= 0 && i < src.cols&&
						j >= 0 && j < src.rows)		//û����
						if (src.at<uchar>(j, i)>200)
						{
							bdps_.push_back(cv::Point(i, j));
							src.at<uchar>(j, i) = 0;	//��Ӹõ������
						}
		}
		//��һ�ֱ߽�Ϊ0ʱ�˳���
		if (bdps_.size() == 0)
			return;
		//����һ�ֱ߽����жϣ��Ƿ���չ������������
		int ps_nb_size0 = ps_nb.size();	//�ڵ���ʼ����
		std::vector<cv::Point>::iterator
			it_ = bdps_.begin(),
			ite_ = bdps_.end();
		for (; it_ != ite_; it_++)
		{
			std::vector<Point>::iterator itc = ps.begin(), itce = ps.end();
			for (; itc != itce; itc++)
			{
				if (*itc == *it_)
				{
					ps_nb.push_back(*itc);
				}
			}
		}
		int ps_nb_size1 = ps_nb.size();		//�жϺ��ڵ�����
		if ((ps_nb_size1 - ps_nb_size0) != 0)	//��չ������������
		{
			//�������ڵ�,�ж���߽�ľ��룬��ȥ����ı߽�
			for (int i = ps_nb_size0; i < ps_nb_size1; i++)	
			{
				int n_ps_nb_x = ps_nb[i].x;
				int n_ps_nb_y = ps_nb[i].y;
				int minx = n_ps_nb_x - kerDist, maxx = n_ps_nb_x + kerDist;
				int miny = n_ps_nb_y - kerDist, maxy = n_ps_nb_y + kerDist;
				std::vector<cv::Point>::iterator
					j_it = bdps_.begin();
				for (; j_it != bdps_.end();)		//�����߽�
				{
					int x_ = j_it->x, y_ = j_it->y;
					if (x_ >= minx&&x_ <= maxx&&
						y_ >= miny&&y_ <= maxy)		//�߽��������������
						j_it = bdps_.erase(j_it);
					else
						j_it++;
				}
			}
		}
		if (bdps_.size() == 0)	//��ȥ�����㸽���ı߽�����Ϊ���˳�
			break;
		bdps = bdps_;
	}
}

//Point�㼯ת��ΪCCPoint�㼯
void ps2cps(Mat* pic, std::vector<cv::Point> ps, std::vector<CCPoint> &cps)
{
	cps.clear();

	//����ȱʡֵ
	int ind = -1;
	int out_dgr = -1;
	std::vector<cv::Point> p_nb;

	std::vector<cv::Point>::iterator psit = ps.begin(), psite = ps.end();
	for (; psit != psite; psit++)
	{
		p_nb.clear();
		neighbor_ps(pic, *psit, ps, p_nb);
		out_dgr = p_nb.size();
		CCPoint cptemp = { ind, out_dgr, *psit, p_nb };
		cps.push_back(cptemp);
	}
}

//��cps���з��ࣨ�����ȣ������ȣ��߳������ࣩ
void classifyByOutdgr(vector<CCPoint> cps, vector<CCPoint> &cps_o1, 
	vector<CCPoint> &cps_o2, vector<CCPoint> &cps_o3)
{
	cps_o1.clear();
	cps_o2.clear();
	cps_o3.clear();
	std::vector<CCPoint>::iterator cpsit = cps.begin(), cpsite = cps.end();
	for (; cpsit != cpsite; cpsit++)
	{
		int outd = cpsit->out_dgr;
		if (outd == 0 ||
			outd == 1)		//�����Ȼ������
			cps_o1.push_back(*cpsit);
		else if (outd == 2)
			cps_o2.push_back(*cpsit);
		else
			cps_o3.push_back(*cpsit);
	}
}

int dist_2p(int x, int y, CCPoint cp)
{
	int x_ = cp.p.x;
	int y_ = cp.p.y;

	return (x - x_)*(x - x_) + (y - y_)*(y - y_);
}

void del_cp(vector<CCPoint> &cps, CCPoint cp)
{
	vector<CCPoint>::iterator it = cps.begin(), ite = cps.end();
	for (; it != ite; it++)
	{
		if (cp.p == it->p)
		{
			////ֱ��erase��ı�����������ڴ��еĴ洢λ�ã����Բ��ô˷�����
			//cps.erase(it);
			it->ind = -5;	//��-5����ʾ�˵��ѱ�ɾ
			break;
		}
	}
}
void del_p(vector<Point> &ps, Point p)
{
	vector<Point>::iterator it = ps.begin(), ite = ps.end();
	for (; it != ite; it++)
	{
		if (p == *it)
		{
			ps.erase(it);
			break;
		}
	}
}

//���ʱ��һ��
void next_nbcp(CCPoint cp, Point &cp_nt)
{
	/*Ѱ����ˮƽ�߻�ֱ�߼н���С�ĵ�*/
	int x = cp.p.x,	y = cp.p.y;
	int x1 = x + 10, y1 = y;		//x�ᷴ���ӳ�����һ��
	int x2 = x, y2 = y + 10;		//y�ᷴ���ӳ�����һ��
	float c1 = sqrt(pow((x1 - x), 2) + pow((y1 - y), 2));
	float c2 = sqrt(pow((x2 - x), 2) + pow((y2 - y), 2));
	float dgr = 6.29;	//�н�
	vector<cv::Point>::iterator it = cp.p_nb.begin(), ite = cp.p_nb.end();
	for (; it != ite; it++)
	{
		int x_ = it->x - x;
		int y_ = it->y - y;
		float a = sqrt(pow((x_ - x), 2) + pow((y_ - y), 2));
		float b1 = sqrt(pow((x_ - x1), 2) + pow((y_ - y1), 2));
		float b2 = sqrt(pow((x_ - x2), 2) + pow((y_ - y2), 2));
		float dgr1 = abs(acos((pow(a, 2) + pow(c1, 2) - pow(b1, 2)) / (2.0*a*c1)));
		float dgr2 = abs(acos((pow(a, 2) + pow(c2, 2) - pow(b2, 2)) / (2.0*a*c2)));
		float dgr_ = min(dgr1, dgr2);
		if (dgr_ < dgr)
		{
			dgr = dgr_;
			cp_nt = *it;
		}
	}
}
//;����ʱ��һ�㣬��һ�㡢���㡢��һ�㣬����trueʱ���Լ����ߣ�����falseʱֹͣ
bool next_nbcp(CCPoint cp1, CCPoint cp2, Point &cp_nt)
{
	/* */
	int x1 = cp1.p.x, y1 = cp1.p.y;
	int x2 = cp2.p.x, y2 = cp2.p.y;
	float c = sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2));
	float dgr = 6.28;	//�н�
	vector<cv::Point>::iterator it = cp2.p_nb.begin(), ite = cp2.p_nb.end();
	for (; it != ite; it++)
	{
		int x = it->x;
		int y = it->y;
		float b = sqrt(pow((x1 - x), 2) + pow((y1 - y), 2));
		float a = sqrt(pow((x - x2), 2) + pow((y - y2), 2));
		float dgr_ = abs(acos((pow(b, 2) + pow(c, 2) - pow(a, 2)) / (2.0*b*c)));

		if (dgr_ < dgr)
		{
			dgr = dgr_;
			cp_nt = *it;
		}
	}
	if (/*(cp_nt.y < y2) ||*/ ((cp_nt.x - x2 + cp_nt.y - y2) >= 0))	//��һ�����ߣ��������ϰ�ߣ�
		return true;
	else
		return false;
}

int realSize(vector<CCPoint> cps)
{
	int sz = 0;
	vector<CCPoint>::iterator it = cps.begin(), ite = cps.end();
	for (; it != ite; it++)
	{
		if (it->ind != -5)
			sz++;
	}
	return sz;
}

//�ʻ��滮
void stroke_plan(Mat* pic, vector<CCPoint> cps, vector<vector<Point>> &spp)
{
	std::vector<cv::Point> sp;		//һ�ʻ�
	int xx = 10000, yy = 10000;		//�����ھ����������Ͻ�xy����ֵ
	
	//ȷ��xx��yy
	vector<CCPoint>::iterator itxy = cps.begin(), itxye = cps.end();
	for (; itxy!=itxye; itxy++)
	{
		int x = itxy->p.x;
		int y = itxy->p.y;
		if (x < xx)
			xx = x;
		if (y < yy)
			yy = y;
	}

	//����������ĵ㿪ʼд
	CCPoint* cp_now = &(cps[0]);	//ȱʡֵΪ��һ��
	while (1)	//ÿһ��ѭ���������㵽�յ��һ���ʻ�
	{
		//���㼯�Ƿ�Ϊ��
		int rsize_ = realSize(cps);
		if (rsize_ <= 0)
			break;

		//Ѱ����ʵ�(����������ĵ�)cp_now
		int dist = 100000000;
		vector<CCPoint>::iterator it = cps.begin(), ite = cps.end();
		for (; it != ite; it++)
		{
			if (it->ind != -5)
			{
				int dist_ = dist_2p(xx, yy, *it);
				if (dist_ < dist)
				{
					cp_now = &(*it);
					dist = dist_;
				}
			}
		}

		//��������ж���һ��
		sp.clear();		//��һ�����
		if (cp_now->out_dgr <= 0)	//����Ϊ�㣬û����һ����
		{
			sp.push_back(cp_now->p);
			del_cp(cps, *cp_now);
			spp.push_back(sp);
			continue;		//����ѭ����һ��
		}	//����д���Ȳ�Ϊ��ʱ
		sp.push_back(cp_now->p);		//��ӵ�һ����
		CCPoint* cp_last = cp_now;		
		Point p_next_temp;
		next_nbcp(*cp_now, p_next_temp);
		cp_now = Point2CCPoint(p_next_temp, cps);
		cp_last->out_dgr--;		//�����һ�����ȼ�һ��
		if (cp_last->out_dgr <= 0)	//��һ�����Ϊ����ɾ���õ�
			del_cp(cps, *cp_last);
		else
			del_p(cp_last->p_nb, p_next_temp);		//�ڵ��һ

		//��һ��һֱѭ��д��ȥ��ֱ������Ϊ����򲻶�
		while (true)	//ÿ��ѭ�������һ�㵽�ڵ��һ��
		{
			sp.push_back(cp_now->p);		//�����һ��
			cp_now->out_dgr--;		//��ȥ�󣬳��ȼ�һ��
			if (cp_now->out_dgr <= 0)	//��һ�����Ϊ����ɾ���õ�
			{
				del_cp(cps, *cp_now);
				spp.push_back(sp);
				break;	//��һ�ʽ��� ����
			}
			else
			{
				del_p(cp_now->p_nb, cp_last->p);		//�ڵ��һ
			}
			if (next_nbcp(*cp_last, *cp_now, p_next_temp))	//���¿���д
			{
				cp_last = cp_now;
				cp_now = Point2CCPoint(p_next_temp, cps);
				cp_last->out_dgr--;
				if (cp_last->out_dgr <= 0)	//��һ����ȼ�һ�����Ϊ����ɾ���õ�
					del_cp(cps, *cp_last);
				else
					del_p(cp_last->p_nb, p_next_temp);
			}
			else	//���²���д�ˣ����ܷ��򲻶ԣ�
			{
				spp.push_back(sp);
				break;	//��һ�ʽ��� ����
			}
		}
	}
}

void insideSetTo0(Mat *pic)
{
	int r = pic->rows, c = pic->cols;
	for (int i = 1; i < r - 1; i++)
	{
		uchar* data = pic->ptr<uchar>(i);
		data++;
		for (int j = 1; j < c - 1; j++)
		{
			*data++ = 0;
		}
	}
}

void multiout_dtct()
{
	//����
	int kern = 3;

	//dst
	std::vector<cv::Point> ps;

	std::vector<std::vector<CvPoint>> rs;
	cv::Mat kernel;
	//string pfn = "D:\\csimg\\hanzijiaodian\\ph_xihua\\xihuaOut2\\2.jpg";
	string pfn = "D:\\csimg\\hanzijiaodian\\ph_xihua\\xihuaOut2\\xihuaOut2\\8.jpg";
	cv::Mat src = cv::imread(pfn, 0);
	//cv::Mat pic = cv::imread("D:\\csimg\\hanzijiaodian\\jing.bmp", 0);
	cv::Mat pic;
	cv::threshold(src, pic, 100, 255, cv::ThresholdTypes::THRESH_BINARY);	//��ֵ��
	//cv::imshow("erzhihua", pic);
	int r = pic.rows, c = pic.cols;
	for (int i = 0; i < r; i++)
	{
		uchar* data = pic.ptr<uchar>(i);
		for (int j = 0; j < c; j++)
		{
			//�����ںڵװ��������
			if (*data++ > 200 &&
				i >= kern / 2 && i <= (r - kern / 2 - 1) &&
				j >= kern / 2 && j <= (c - kern / 2 - 1))			//�߽�ͺ�ɫ���������ж�
			{
				rs.clear();
				/*for (int i_ = i - kern / 2; i_ <= i + kern / 2; i_++)
				for (int j_ = j - kern / 2; j_ <= j + kern / 2; j_++)
				{
				if ((i_ == i - kern / 2) ||
				(i_ == i + kern / 2) ||
				(j_ == j - kern / 2) ||
				(j_ == j + kern / 2))
				kernel.at<uchar>(j_, i_) = pic.at<uchar>(j_, i_);
				}*/
				pic(Rect2i(j - kern / 2, i - kern / 2, kern, kern)).copyTo(kernel);
				insideSetTo0(&kernel);
				az::findconnectedregions(&kernel, rs);
				if (rs.size() == 1)	//�жϳ���
					ps.push_back(Point(j, i));
			}
		}
	}

	cout << ps.size() << endl;
	cv::cvtColor(pic, pic, CV_GRAY2BGR);
	std::vector<cv::Point>::iterator it = ps.begin();
	for (; it != ps.end(); it++)
	{
		//circle(pic, *it, 3, Scalar(0, 0, 255), 2);
		pic.at<cv::Vec3b>(*it) = Vec3b(0, 0, 255);
	}
	cv::imshow(pfn, pic);
	cv::waitKey();
}

////////////////�ϲ�����ȡ�ʻ����滮˳���²���ģ��д��///////////////////

void get_TwoPs(vector<vector<Point>> spp, vector<pair<Point, Point>> &tps)
{
	tps.clear();
	pair<Point, Point> pairpp;
	vector<vector<Point>>::iterator it = spp.begin(), ite = spp.end();
	for (; it != ite; it++)
	{
		int itsize = it->size();
		for (int i = 0; i < itsize; i++)
		{
			if (i != itsize - 1)	//����β��
			{
				pairpp.first = it->at(i);
				pairpp.second = it->at(i + 1);
				tps.push_back(pairpp);
			}
		}
	}
}

void tp2tps(pair<Point, Point> tp, vector<pair<Point, Point>> &tps)
{
	//parm
	float sp = 5.0;

	//ģ�ⶶ��
	//��ȡ�����xr,yr(��Χ-writeShakeRange��writeShakeRange)
	int xr, yr, xr2, yr2;
	if (writeShakeRange != 0)
	{
		srand(getTime());
		xr = rand() % (writeShakeRange * 2);
		xr -= writeShakeRange;
		xr = xr < 0 ? 0 : xr;
		xr = xr > pic_w ? pic_w : xr;
		srand(getTime() + 191);		//������
		yr = rand() % (writeShakeRange * 2);
		yr -= writeShakeRange;
		yr = yr < 0 ? 0 : yr;
		yr = yr > pic_h ? pic_h : yr;
		//srand(getTime() + 373);
		//xr2 = rand() % (writeShakeRange * 2);
		//xr -= writeShakeRange;
		//srand(getTime() + 499);
		//yr2 = rand() % (writeShakeRange * 2);
		//yr -= writeShakeRange;
		xr2 = 0, yr2 = 0;
	}
	else
	{
		xr = 0, yr = 0;
		xr2 = 0, yr2 = 0;
	}
	float x1 = tp.first.x + xr, y1 = tp.first.y + yr;
	float x2 = tp.second.x + xr2, y2 = tp.second.y + yr2;
	float dis = sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
	float num = dis / sp;		//�ֽ�ɶ��ٷ�

	tps.clear();
	pair<Point, Point> smtp;
	for (int i = 0; i <= num; i++)
	{
		smtp.first = Point((int)(x1 + i / num*(x2 - x1)), (int)(y1 - i / num*(y1 - y2)));
		smtp.second = Point((int)(x1 + (i + 1) / num*(x2 - x1)), (int)(y1 - (i + 1) / num*(y1 - y2)));
		tps.push_back(smtp);
	}
}

void write(vector<vector<Point>> spp, cv::Size s, std::vector<cv::Point> ps)
{
	cv::Mat pic(s, CV_8UC3);
	pic.setTo(Scalar(255, 255, 255));
	vector<pair<Point, Point>> tps;
	vector<pair<Point, Point>> smtps;
	get_TwoPs(spp, tps);
	cv::namedWindow("write");
	cv::moveWindow("write", 300, 350);
	cv::imshow("write", pic);

	//��ʼģ��д��
mk_beginWrite:
	int ind = 1;
	int sb = 0;
	cv::waitKey(300);
	int wsr = writeShakeRange;
	Point p_last(-10, -10);
	vector<pair<Point, Point>>::iterator it = tps.begin(), ite = tps.end();
	for (; it != ite; it++)
	{
		if (it->first == p_last)		//�±ʵ��������
			writeShakeRange = 0;
		else
			writeShakeRange = wsr;
		tp2tps(*it, smtps);
		vector<pair<Point, Point>>::iterator smit = smtps.begin(), smite = smtps.end();
		for (; smit != smite; smit++)
		{
			cv::line(pic, smit->first, smit->second, Scalar(255, 0, 0),
				writeThickness/*, LineTypes::LINE_AA*/);
			cv::imshow("write", pic);
			if (saveIt && sb++ % 3 == 0)
			{
				string path = "D:\\csimg\\hanzijiaodian\\д�ֹ���\\" + to_string(ind++) + ".bmp";
				cv::imwrite(path, pic);
			}
			cv::waitKey(spd1);
		}
		cv::waitKey(spd2);
		p_last = it->second;
	}

	//���潻����ʾ���
	cv::Mat pic0;
	int k = 0;
	pic.copyTo(pic0);
	std::vector<cv::Point>::iterator itp = ps.begin(), itpe = ps.end();
	for (; itp != itpe; itp++)
	{
		cv::circle(pic, *itp, 3, Scalar(0, 0, 255), 3);
	}
	if (saveIt)
	{
		string path = "D:\\csimg\\hanzijiaodian\\д�ֹ���\\" + to_string(ind++) + ".bmp";
		cv::imwrite(path, pic);
	}
	cv::waitKey(500);
	char c;
	while (true)
	{
		if (k++ % 2 == 0)
		{
			cv::imshow("write", pic);
			c = cv::waitKey(800);
			if (c == 'r')		//��дһ��
			{
				pic.setTo(Scalar(255, 255, 255));
				goto mk_beginWrite;
			}
		}
		else
		{
			cv::imshow("write", pic0);
			c = cv::waitKey(800);
			if (c == 'r')		//��дһ��
			{
				pic.setTo(Scalar(255, 255, 255));
				goto mk_beginWrite;
			}
		}
	}
	cv::waitKey();
}

//�����ѻ�ȡ��˳��orderly_ps������ͼ���е��������Ӧ����ӳ�䵽���к����ϣ��滮������
void zzzz(char* img_path, vector<Point> orderly_ps, vector<BEPoint> &dst)
{
	//���
	dst.clear();

	//��cps
	Mat picsrc = imread(img_path, 0);
	cv::threshold(picsrc, picsrc, 128, 1, cv::THRESH_BINARY_INV);
	Mat pic = thinImage(picsrc);
	pic *= 255;
	pic_w = pic.cols;
	pic_h = pic.rows;
	std::vector<cv::Point> ps;
	harris_dtct(pic, ps);
	zhengli_ps(&pic, ps);
	std::vector<CCPoint> cps;
	ps2cps(&pic, ps, cps);

	//

}

/////////////////������룺��Ҫ����յĲ���̫�ݵĵ����ȡ����/////////////////

//�����������е�ķ���ֵ:ͼ����ʼ�㣬�յ㣬���������
void jisuanfangxiangzhi(Mat *img, Point st, Point ed, int pr, vector<pair<Point, int>> &pada)
{
	Mat pic;
	img->copyTo(pic);
	vector<Point> qu;
	qu.clear();
	pada.clear();
	int radius = pr / 2;
	while (1)
	{
		qu.push_back(st);	pic.at<uchar>(st) = 0;
		if (pic.at<uchar>(st.y, st.x - 1) == 255)
			st = Point(st.x - 1, st.y);
		else if (pic.at<uchar>(st.y, st.x + 1) == 255)
			st = Point(st.x + 1, st.y);
		else if (pic.at<uchar>(st.y - 1, st.x) == 255)
			st = Point(st.x, st.y - 1);
		else if (pic.at<uchar>(st.y + 1, st.x) == 255)
			st = Point(st.x, st.y + 1);
		else if (pic.at<uchar>(st.y - 1, st.x - 1) == 255)
			st = Point(st.x - 1, st.y - 1);
		else if (pic.at<uchar>(st.y + 1, st.x + 1) == 255)
			st = Point(st.x + 1, st.y + 1);
		else if (pic.at<uchar>(st.y - 1, st.x + 1) == 255)
			st = Point(st.x + 1, st.y - 1);
		else if (pic.at<uchar>(st.y + 1, st.x - 1) == 255)
			st = Point(st.x - 1, st.y + 1);
		else
			break;
		if (st == ed)
			break;
		int qus = qu.size();
		if (qus >= pr)//�����м��ķ���ֵ,��λ���ȣ���-90~90
		{
			int x2 = qu[qus - 1].x;
			int y2 = qu[qus - 1].y;
			int x1 = qu[qus - pr].x;
			int y1 = qu[qus - pr].y;
			//��Ϊͼ������ϵ�����Ǻ�������ϵ��һ���������������һ������
			int du = -(atan((float)(y2 - y1) / (float)(x2 - x1)))*(180. / 3.1416);
			pada.push_back(pair<Point, int>(qu[qus - 1 - radius], du));
		}
	}
}

//�����������е�ķ���ֵ:ͼ����ʼ�㣬�յ㣬���������
void jisuanfangxiangzhi2(vector<Point> bh, int pr, vector<pair<Point, int>> &pada)
{
	pada.clear();
	int radius = pr / 2;
	int si = bh.size();
	Point ed = bh[si - 1];
	for (int i = 0; i < si; i++)
	{
		if (i >= pr)
		{
			int x2 = bh[i].x;
			int y2 = bh[i].y;
			int x1 = bh[i - pr].x;
			int y1 = bh[i - pr].y;
			//��Ϊͼ������ϵ�����Ǻ�������ϵ��һ���������������һ������
			int du = -(atan((float)(y2 - y1) / (float)(x2 - x1)))*(180. / 3.1416);
			pada.push_back(pair<Point, int>(bh[i - radius], du));
		}
	}
}

//�������淽��ֵ���㲻̫�յĵ�,par_kd:�ȽϵĿ�ȣ�par_jd���ȽϵĽǶ���ֵ
void jisuanguaidian(vector<pair<Point, int>> pada, int par_kd,int par_jd, vector<Point> &ps)
{
	int dif(int a, int b);

	//ps.clear();
	int size = pada.size();
	int *du = new int[size];
	int k = 0;
	vector<pair<Point, int>>::iterator itb = pada.begin(), ite = pada.end();
	for (; itb != ite; itb++)
	{
		du[k++] = itb->second;
	}

	//��һ������ȡ������ֵ�仯�ĵ㼯
	vector<Point> pstemp;
	for (int i = par_kd - 1; i < size - (par_kd - 1); i++)
	{
		if (dif(du[i], du[i - (par_kd - 1)]) >= par_jd||//"��"��"��"�����Ը����������
			dif(du[i], du[i + (par_kd - 1)]) >= par_jd)//������ֵ,ǰ�󶼱Ƚ�
			pstemp.push_back(pada[i].first);
	}

	//�ڶ��������ݵ㼯���м��
	vector<Point> vp;//��������һ����������
	vector<Point>::iterator itb2 = pstemp.begin(), ite2 = pstemp.end();
	for (; itb2 != ite2; itb2++)
	{
		int ss = vp.size();
		if (ss == 0)
			vp.push_back(*itb2);
		else
		{
			Point p1 = vp[ss - 1];
			Point p2 = *itb2;
			if (abs(p1.x - p2.x) <= 5 &&
				abs(p1.y - p2.y) <= 5)	//////////////////////////��ֵ��7�������ߵľ�����ֵ
				vp.push_back(p2);
			else	///�����߽����������������ߵ��е�
			{
				if (ss > 5)	//////////////////����С��5������������������
					ps.push_back(vp[(int)(ss / 2)]);
				vp.clear();	//�����߽���������
			}
		}
	}
	//��������ٴ��ж�
	if (vp.size() > 5)	//////////////////����С��5������������������
		ps.push_back(vp[(int)(vp.size() / 2)]);

	delete[] du;
}

//��������ǵĲ-90~90��Χ
int dif(int a, int b)
{
	int dif1 = abs(a - b);
	int dif2 = abs((90 - a) + (-90 - b));
	int dif3 = abs((90 - b) + (-90 - a));

	return min((min(dif1, dif2)), dif3);
}

////////////////����cps���ԣ�һ���м�����������ߵ������˵㣩
void cps2lps(vector<CCPoint> cps, vector<pair<Point, Point>> &pps)
{
	vector<CCPoint>::iterator itb = cps.begin(), ite = cps.end();
	for (; itb != ite; itb++)
	{
		Point p0 = itb->p;
		Point p1;
		int ss = itb->p_nb.size();
		for (int i = 0; i < ss; i++)
		{
			p1 = itb->p_nb[i];
			//Ϊ��ֹ�ظ�������һ������ֵ֮�ʹ�С���ж�
			if ((p0.x + p0.y) <= (p1.x + p1.y))
				pps.push_back(pair<Point, Point>(p0, p1));
		}
	}
}

//����һ��ͼ�������㣬�����бʻ�
void getAllBihua(Mat *img, vector<Point> ps, vector<vector<Point>> &bhs)
{
	vector<Point> bh;
	Mat pic;
	img->copyTo(pic);
	int sz = ps.size();
	for (int i = 0; i < sz; i++)
	{
		pic.at<uchar>(ps[i]) = 0;
		circle(pic, ps[i], 5, Scalar(0, 0, 0), 7);
	}
	//imwrite("D:\\csimg\\���ֲ���̫�յĵ���ȡ\\temp1.bmp", pic);
	vector<Point> oneout_ps;
	oneout_dtct(pic, oneout_ps);
	int si = oneout_ps.size();
	for (int i = 0; i < si; i++)
	{
		bh.clear();
		pot_erg(&pic, oneout_ps[i], bh);
		if (bh.size() > 10)
			bhs.push_back(bh);
	}
}

//����,һ����λ������
void getAllFeaturePoints(Mat *img, int par_bc, int par_kd, int par_jd, vector<Point> &rps)
{
	vector<pair<Point, int>> pada;
	std::vector<cv::Point> ps;//haaris��������������
	std::vector<cv::Point> gds;//����ר�Ŵ�Ų���̫�յĵ�
	vector<vector<Point>> bhs;//���еıʻ�

	harris_dtct(*img, ps);
	zhengli_ps(img, ps);
	getAllBihua(img, ps, bhs);
	
	//����̫�յĵ�
	vector<vector<Point>>::iterator itb = bhs.begin(), ite = bhs.end();
	for (; itb != ite; itb++)
	{
		pada.clear();
		jisuanfangxiangzhi2(*itb, par_bc, pada);
		jisuanguaidian(pada, par_kd, par_jd, gds);
	}

	//���������㼯
	//rps.insert(rps.end(), ps.begin(), ps.end());
	//rps.insert(rps.end(), gds.begin(), gds.end());
	merge_ps_2(img, ps, gds, rps);
	//merge_ps(ps, gds, rps, img->rows, img->cols);
	zhengli_ps(img, rps);
}

//////////����һ���㣬��ȱ�������ͨ����,ʹ��֮ǰ�ȿ���һ��img
void getGdLTQY(Mat *img, vector<Point> qidian, vector<Point> &ps)
{
	vector<Point> bianjie;
	bianjie.clear();
	//�жϱ߽����Ƿ��������㣬�����浽ps
	vector<Point>::iterator itb = qidian.begin(), ite = qidian.end();
	for (; itb != ite; itb++)
	{
		if (img->at<uchar>(*itb) != 255)//����������
			return;
		img->push_back(*itb);
	}
	//���µı߽�,��Ϊ0������
	itb = qidian.begin();
	for (; itb != ite; itb++)
	{
		img->at<uchar>(*itb) = 0;
		//��9�ո��б���
		for (int i = itb->x - 1; i <= itb->x + 1; i++)
		{
			for (int j = itb->y - 1; j <= itb->y + 1; j++)
			{
				if (i >= 0 && i < img->cols&&
					j >= 0 && j < img->rows)
				{
					if (img->at<uchar>(j, i) != 0)
					{
						img->at<uchar>(j, i) = 0;
						bianjie.push_back(Point(i, j));
					}
				}
			}
		}
	}
	if (bianjie.size() == 0)
		return;
	else
		getGdLTQY(img, bianjie, ps);
}
