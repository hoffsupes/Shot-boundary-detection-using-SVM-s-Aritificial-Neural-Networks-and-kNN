
#include <opencv2/highgui/highgui.hpp>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <vector>
#include <cstdarg>
#include "opencv2/opencv.hpp"
#include "fstream"
#include <dirent.h>
#include <math.h>
#include <time.h>
#include <opencv2/features2d.hpp>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

using namespace std;
using namespace cv;
using namespace cv::ml;

std::ofstream ff;

string get_string(double sha,double va,double v,double h,double d1,double d2)
{
stringstream sa,vaa,vo,ho,d1o,d2o;
sa << sha;
vaa << va;
vo << v;
ho << h;
d1o << d1;
d2o << d2;

string sh("\n SE: ");
sh = sh + sa.str() + "\n Std: " + vaa.str() + "\n H: " + ho.str() + "\n V: " + vo.str() + "\n D1: " + d1o.str() + "\n D2: " + d2o.str() ;

return sh;
}

void disp_diff(double sha,double va,double v,double h,double d1,double d2,Mat frac,Mat fran,Rect ROI)       // used in initial stages
{

cout <<" \n Shannons Entropy difference: "<< sha <<"\n StdDev:: "<< va <<"\n Vertical Edge Difference:: "<< v <<"\n Horizontal Edge difference::"<< h<< "\n";

cout <<" \n Diagonal1 Edge Difference:: "<< d1 <<"\n Diagonal 2 Edge difference::"<< d2<< "\n";

string sh =  get_string(sha,va, v, h, d1, d2);

//putText(frac,sh.data(),ROI.tl(),FONT_HERSHEY_SIMPLEX,0.2,Scalar(0,255,255),0.5);
//putText(fran,sh.data(),ROI.tl(),FONT_HERSHEY_SIMPLEX,0.2,Scalar(0,255,255),0.5);

// imshow("Current Frame",frac);
// imshow("Next Frame",fran);
// waitKey(25);
}

double get_small_feat(Mat blok, double & var)
{
	Mat P;
	int histSize = 256;
	float range[] = { 0, 256 } ;
	const float* histRange = { range };
	calcHist( &blok, 1, 0, Mat(), P, 1, &histSize, &histRange, true, false );
	P.convertTo(P,CV_64F);	
	P = P/ (blok.rows * blok.cols);

	Mat log2p;
	log(P,log2p);

	for(int i = 0; i < log2p.rows; i++){ if(isnan(log2p.at<double>(i))) {log2p.at<double>(i) = 0;} if(isinf(log2p.at<double>(i))) {log2p.at<double>(i) = 0;} }
	log2p /= std::log(2);

	Mat m; Mat vv;
	meanStdDev(blok,m,vv); var = vv.at<double>(0);	
	multiply(P,log2p,log2p);
	return -sum( log2p ).val[0];

}


double get_histo_feat(Mat currf, Mat nextf)
{ 
    	Mat P1,P2;
	int histSize = 256;
	float range[] = { 0, 256 } ;
	const float* histRange = { range };
	calcHist( &currf, 1, 0, Mat(), P1, 1, &histSize, &histRange, true, false );
	calcHist( &nextf, 1, 0, Mat(), P2, 1, &histSize, &histRange, true, false );
        
        normalize(P1,P1,0,1,NORM_MINMAX);
        normalize(P2,P2,0,1,NORM_MINMAX);
    
        return compareHist(P2,P1,CV_COMP_CHISQR);
        
}

Mat get_labels(string file)
{
Mat labmat;
FileStorage fp(file.data(), FileStorage::READ);

fp["labels"] >> labmat;

fp.release();
return labmat;
}

int get_label(int fri, Mat labs)
{
Mat l = labs.col(0);
Mat prev_dat = labs.col(1);
Mat next_dat = labs.col(2);
Mat p;
bitwise_and( (fri >= prev_dat),(fri <= next_dat),p );

if (!countNonZero(p))
{
return 4;
}

Mat k;
findNonZero(p,k);

return l.at<int>( k.at<int>(1) );

}


int get_DLIMIT(Mat labs) // dynamically calculate DLIMIT    // Using dlimit value of 600, for very large video databases, DLIMIT can be automatically calculated using this
{
return (sum(labs.col(2) - labs.col(1) + 1)[0])/ 3;
}


Mat obtain_diff_feat( Mat fracurr, Mat franext,int testmode = 0 , int n = 80 , double vth = 1000, double hth = 1000, double d1th = 1000, double d2th = 1000, double shth = 0.2, double varth = 1.5)		// assuming frames are resied to 360 x 640
{

Rect shifROI = Rect(0,0,n,n);
double shf1, shf2, vaf1,vaf2;
double histi = 0;
double chisti = 0;
double emd = 0;

int M = fracurr.rows, N = fracurr.cols;
Mat features;
int cnt = 0;

	if(!features.empty())
	{
		cout<<"\nFeatures are not empty!! May be due to garbage values!! Exiting\n";
		exit(1);
	}

	for(int mm = 0; mm < (M-n); mm += n)
	{
		for(int nn = 0; nn < (N-n); nn += n)			// test for real threshold values later on
		{

		Mat feat;
                Mat tmp;
                double chii;
                Mat b1,b2, sg1,sg2;
                shifROI = Rect(nn,mm,n,n);

		Mat blokcurr = fracurr(shifROI);
		Mat bloknext = franext(shifROI);		
                chii = get_histo_feat(blokcurr, bloknext);
                chisti += chii;
                shf1 =  get_small_feat(blokcurr, vaf1);
                shf2 =  get_small_feat(bloknext, vaf2);
                
                feat.push_back(shf2 - shf1);
		feat.push_back(vaf2 - vaf1);
                feat.push_back(abs(mean(bloknext)[0] - mean(blokcurr)[0]));
                emd += norm(blokcurr,bloknext,NORM_L2);
                
		if(testmode)
		{
//		disp_diff((shf1 - shf2),abs(vaf1 - vaf2),norm(vf1,vf2,NORM_L2),norm(hf1,hf2,NORM_L2),norm(d1f1,d1f2,NORM_L2),norm(d2f1,d2f2,NORM_L2),fracurr,franext,shifROI);
		}

		if(features.empty())
		{
			features = Mat::zeros(feat.size(), feat.type());
			features += feat;
                        cnt++;
		}
		else
		{
			features += feat;	
                        cnt++;
		}

		}
	}
        features.push_back(chisti / cnt);
        features.push_back(( mean( franext )[0] - mean( fracurr )[0] ));
        features.push_back(emd / cnt);
        
        if(testmode){
        	cout << "\tfeatures:::" << " shannons:::" << features.row(0)<<"\n" << " variance:::" <<  features.row(1)<<"\n" << " SSM::::" << features.row(2) <<"\n" <<  " hdff::: " << features.row(3)<< " manhattan overall "<< features.row(4)<<  "chi :::" << features.row(5)<< "\n template:: " << features.row(6)<< "\n EMD:: " << features.row(7)<< "\n Mnorm:: "<< features.row(8)<<"\n Cnorm:: "<< features.row(9) << "\n Average Change in Means: "<< features.row(10);
        ff << "\tfeatures:::" << " shannons:::" << features.row(0)<<"\n" << " variance:::" <<  features.row(1)<<"\n" << " SSM::::" << features.row(2) <<"\n" <<  " hdff::: " << features.row(3)<< " manhattan overall "<< features.row(4)<<  "chi :::" << features.row(5)<< "\n template:: " << features.row(6)<< "\n EMD:: " << features.row(7)<< "\n Mnorm:: "<< features.row(8)<<"\n Cnorm:: "<< features.row(9) << "\n Average Change in Means: "<< features.row(10);
            // 	cout << "\tfeatures:::" << " shannons:::" << features.row(0)<<"\n" << " variance:::" <<  features.row(1)<<"\n" << " SSM::::" << ssm <<"\n" <<  " hdff::: " << hdff  << " manhattan overall "<< m1 <<  "chi :::" << chi1 << "\n template:: " << templ << "\n template N:: " << templ / cnt << "\n EMD:: " << emd / cnt<< "\n Mnorm:: "<< histi <<"\n Cnorm:: "<< chisti / cnt << "\n Average Change in Means: "<< ( mean( franext )[0] - mean( fracurr )[0] );
//         
//         	ff << "\tfeatures:::" << " shannons:::" << features.row(0)<<"\n" << " variance:::" <<  features.row(1)<<"\n" << " SSM::::" << ssm <<"\n" <<  " hdff::: " << hdff  << " manhattan overall "<< m1 <<  "chi :::" << chi1 << "\n template:: " << templ << "\n template N:: " << templ / cnt << "\n EMD:: " << emd / cnt<< "\n Mnorm:: "<< histi <<"\n Cnorm:: "<< chisti / cnt << "\n Average Change in Means: "<< ( mean( franext )[0] - mean( fracurr )[0] );
        }
        
        features.convertTo(features, CV_32F);
        
return features.t();

}

vector<Mat> get_window(VideoCapture &cap, int i, int M,int N, int cvrt = 1)
{
vector<Mat> fra;
Mat zrr = Mat::zeros(Size(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT)), CV_8UC1);

	if(i < ((N-1)/2) )
	{
	Mat fr;
	
	for(int ii = 0; ii<  ((N-1)/2) - i; ii++ ) 
	{
	fra.push_back(zrr);
	}

	cap.set(CV_CAP_PROP_POS_FRAMES,0);

	while(cap.get(CV_CAP_PROP_POS_FRAMES) <= (i+ ((N-1)/2) ) )
	{
	Mat y;
	cap >> y; if(cvrt){cvtColor(y,y,CV_BGR2GRAY); }
	fra.push_back(y);
	}

	cap.set(CV_CAP_PROP_POS_FRAMES,i+1);

	}
	
	else if(i > M - ((N-1)/2) )
	{

	cap.set(CV_CAP_PROP_POS_FRAMES,i - ((N-1)/2));
	
	while(cap.get(CV_CAP_PROP_POS_FRAMES) < cap.get(CV_CAP_PROP_FRAME_COUNT))
	{
	Mat y;
	cap >> y; if(cvrt){cvtColor(y,y,CV_BGR2GRAY); }
	fra.push_back(y);
	}		
			
	for(int j = 0; j < ((N-1)/2) - (M - i); j++)
	{
	fra.push_back(zrr);	
	}	

	cap.set(CV_CAP_PROP_POS_FRAMES,i+1);
	}

	else if( (i>=((N-1)/2))&&(i <= M - ((N-1)/2)))
	{
	cap.set(CV_CAP_PROP_POS_FRAMES,i-((N-1)/2));
	
	while(cap.get(CV_CAP_PROP_POS_FRAMES) <= (i + ((N-1)/2) ) )
	{
	Mat y;
	cap >> y;if(cvrt){cvtColor(y,y,CV_BGR2GRAY); }
	fra.push_back(y);
	}
	
	cap.set(CV_CAP_PROP_POS_FRAMES,i+1);
	}

	else
	{
	cout <<"\n Invalid Video Index!! Exiting Check Video Parser!!! \n";
	exit(1);
	}

return fra;

}

Mat getcompoundfeat(vector<Mat> fralist,int testmode = 0)
{
Mat featu;

	for(int i = 0; i < fralist.size()-1; i++)
	{
	Mat feat = obtain_diff_feat(fralist[i],fralist[i+1],testmode);	
	if(featu.empty()){featu = Mat::zeros(feat.size(), feat.type());}
	hconcat(feat,featu,featu);
	}
	
return featu;

}

Ptr <SVM> get_SVM_model(Mat data, Mat labels)
{
	labels.convertTo(labels, CV_32SC1);
				// This is very inefficient because converting a whole very large matrix to 32F is intensive??

	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::RBF);
	svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,1000, 1e-6) );
	svm->trainAuto( TrainData::create( data, cv::ml::ROW_SAMPLE, labels ), 5,SVM::getDefaultGrid(SVM::C),SVM::getDefaultGrid(SVM::GAMMA) ); 
return svm;

}

void process_video(string filename, string lname, Mat&data, Mat&labels,int dlimiter_on = 0, int DLIMIT = 300, int fram_s = -1, int fram_lim = -1, int testmode = 0, int win_length = 15)	// use automated 
{

    if(dlimiter_on)
    {
        cout << "\n dlimiter is ON only using " << DLIMIT << " frames of video. \n";
    }
    else
    {
        cout << "\n dlimiter is OFF using ALL frames of video. \n";
    }
    
VideoCapture cap;
cap.open(filename.data()); // work on automating reading this file use vector strings
cout << "\n Training Data in :: " << filename.data() << "    training labels in :: "<<lname.data() << "\n";
int dlim = 0;

cout << "frames:: "<< cap.get(CV_CAP_PROP_FRAME_COUNT);

if(!cap.isOpened())
{
    cout << "\n ERROR IN OPENING VIDEO FILE!! SKIPPING VIDEO FILE!! \n";
    return ;
}

Mat lub = get_labels(lname);
int i = 0;

if (fram_s > 0) {cap.set(CV_CAP_PROP_POS_FRAMES,fram_s);}
if (fram_lim < 0) {fram_lim = cap.get(CV_CAP_PROP_FRAME_COUNT);}

while( cap.get(CV_CAP_PROP_POS_FRAMES) < fram_lim )
{
//cout << "\n label :: " << i;
int llb = get_label(i,lub);    

if(llb == 4 && dlimiter_on == 1)
{
    if(dlim == DLIMIT){cout <<"\n DLIMIT reached for normal frames!! \n";}
    if(dlim > DLIMIT)
    {
        i++;
        cap.set(CV_CAP_PROP_POS_FRAMES,i);
        continue;
    }
    dlim++;
}

labels.push_back(llb); 

//if(labels.at<int>(labels.rows - 1) == 4) {cout << "\n NO"; ff << "\n NO";} else {cout << "\n YES";ff << "\n YES";}
if (testmode) {if(labels.at<int>(labels.rows - 1) == 4) {testmode = 1; cout << "\n For frame "<< i <<"\n";}}

////
data.push_back(getcompoundfeat(get_window(cap,i, cap.get(CV_CAP_PROP_FRAME_COUNT) - 1,win_length),testmode));
////

i++;
}

// data.col(data.cols-1) = do_der(data.col(data.cols-1));
// cout << "\n Manhattan :: "<< data.col(data.cols-1) << "\n";
labels.convertTo(labels,CV_32SC1);
// save data and labels later on
cout << "\n Reached here!!! \n";
/*
string filn = filename + "_features.txt";
FileStorage file_p(filn.data(), FileStorage::WRITE);
file_p << data;
file_p.release();


Ptr <SVM> svm = get_SVM_model(data,labels);

string model = filename + "_svmmodel.svm";
svm->save(model.data());
*/
}

string get_filename(string fila)
{
 return fila.substr(fila.find_last_of("/")+1,fila.find_last_of(".") - fila.find_last_of("/") - 1);
}

void generate_data(string trndata,string trainlab,string traind_path = string("Training_Data.xml"),string trainl_path = string("Training_Labels.xml"), int dlimiter_on = 0, int DLIMIT = 330)
{ // call twice for training and testing
	DIR *pdir = NULL;
	pdir = opendir(trndata.data()); 
        
	if(pdir ==  NULL){cout<<"\nFile Directory Inaccessible!!\n";exit(1);}	
	struct dirent *pent = NULL;
	
	FileStorage file_p(traind_path.data(), FileStorage::WRITE);
	file_p << "Data" << "[";
	
	FileStorage file_p2(trainl_path.data(), FileStorage::WRITE);
	file_p2 << "Labels" << "[";
        
	while( pent = readdir(pdir) )
	{

            Mat train_data,train_labels;	
            if(pent == NULL){cout<<"\nCheck Your files / you may not have permission to access this folder. \n"; exit(1);}
            
            string * filnam = new string( pent->d_name );
            if(filnam->at(0) == '.') { continue; }
            
            string t = get_filename(pent->d_name);
            string trlb = trainlab;
            string trdt = trndata;
            
            trlb.append("/");
            trlb.append(t);
            trlb.append(".xml");
            trdt.append("/");
            trdt.append(pent->d_name);
                
            process_video(trdt, trlb,train_data,train_labels,dlimiter_on,DLIMIT);
            
            cout <<"\n Exited proces_video\n";
            file_p << train_data;
            file_p2 << train_labels;
            
            cout << "\n Train Data being written :: "<< train_data.size() << "\n";
            cout << "\n Train Labels being written :: "<< train_labels.size() << "\n";
                
	}
	

        
	file_p2 << "]";
	file_p2.release();
	
	file_p << "]";
	file_p.release();
}

Mat normr(Mat x)
{

for(int i = 0; i < x.rows; i++)
{
    for(int j = 0; j < x.cols; j++)
    {
        if(isinf(x.at<float>(i,j)))
        {
        x.at<float>(i,j) = 0;
        }
        
        if(isnan(x.at<float>(i,j)))
        {
        x.at<float>(i,j) = 0;
        }
    }
}    

Mat xc,k,rd;
x.convertTo(x,CV_64F);
pow(x,2,xc);
reduce(xc,xc,1,REDUCE_SUM);
pow(xc,0.5,xc);
repeat(xc,1,x.cols,k);

divide(x,k,rd);
Mat H = (xc == 0);

if(countNonZero(H))
{
    Mat ruw = Mat::ones(1,x.cols,CV_8UC1) / sqrt(x.cols);
    Mat ps;
    for(int i = 0; i < H.rows; i++)
    {
        if(H.at<uchar>(i) != 0)
        {
            ruw.copyTo(rd.row(i)); 
            
        }

    }

}
rd.convertTo(rd,CV_32F);
return rd;
}

void disp_vecti(vector<int> f)
{
    cout <<"\n[ ";
    for(int i = 0; i < f.size(); i++)
    {
        cout << f[i];
        if (i != f.size() - 1){cout <<" ,";}
    }
    cout << " ]\n";
}

void disp_vectf(vector<float> f)
{
    cout <<"\n[ ";
    for(int i = 0; i < f.size(); i++)
    {
        cout << f[i];
        if (i != f.size() - 1){cout <<" ,";}
    }
    cout << " ]\n";
}

void disp_vectm(vector<Mat> f)
{
    for(int i = 0; i < f.size(); i++)
    {
        cout << i << "   \n";
        cout << f[i] <<"\n";
    }
}

PCA preprocess_data_train(Mat& tdata, Mat & newdata, int feature_sel = 70)
{
    
// Mat k;
// cv::pow(tdata,2,k);
// reduce(k,k,1,REDUCE_SUM);
// cv::sqrt(k,k);
// 
// for(int i = 0; i < tdata.rows; i++)
// {
//     tdata.row(i) = tdata.row(i) / ( k.at<float>(i) ) ;
// }
// 
// tdata = tdata.rowRange(0,50);
// cout << tdata << "\n";
// waitKey();

Mat average;
PCA pca(tdata, average, CV_PCA_DATA_AS_ROW, feature_sel);
newdata = Mat::zeros(tdata.rows, feature_sel,tdata.type());
pca.project(tdata,newdata);

return pca;
}

void preprocess_data_test(Mat& tdata,PCA pca, Mat &newdata,  int feature_sel = 70)
{
    
// Mat k;
// cv::pow(tdata,2,k);
// reduce(k,k,1,REDUCE_SUM);
// cv::sqrt(k,k);
// 
// for(int i = 0; i < tdata.rows; i++)
// {
//     tdata.row(i) = tdata.row(i) / ( k.at<float>(i) ) ;
// }
// 
// tdata = tdata.rowRange(0,50);
// cout << tdata << "\n";
// waitKey();

Mat average;
newdata = Mat::zeros(tdata.rows, feature_sel,tdata.type());
pca.project(tdata,newdata);
}

void precision_recall(Mat l1, Mat l2,vector<float> &pr,vector<float> &re)
{
    double correct = countNonZero(l1 == l2);
    double missing = countNonZero((l1!=4) == (l2==4));
    double fals = countNonZero((l1==4) == (l2!=4));
    
    cout << "\n Precision is :: "<< (correct)/(correct + fals);
    cout << "\n Recall is :: "<< (correct)/(correct + missing);
    
    pr.push_back((correct)/(correct + fals));
    re.push_back((correct)/(correct + missing));
    
}

Mat confusion_mat(Mat alab, Mat plab, int cla = 5)
{
    Mat conf = Mat::zeros(cla,cla,CV_32F);
    
    for(int i = 0; i < cla; i++)
    {
        for(int j = 0; j < cla; j++)
        {
        Mat l;
        float n = 0;
        bitwise_and((alab == i),(plab == j),l);
        n = countNonZero(l);
        conf.at<float>(i,j) = n;       
        }

    }
    
    return conf;
}
