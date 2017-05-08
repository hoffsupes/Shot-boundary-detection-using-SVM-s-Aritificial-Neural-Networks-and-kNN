
#include "shob_lib.hpp"

Ptr<SVM> train_svm(PCA& pca, vector<float> &tra, int featu = 70, int dlimiter_on = 0, unsigned long DLIMIT = 635)
{
        Mat train_data;
	Mat train_label;
        Mat t_dat, t_lab;
        Mat k;
        unsigned long dlim = 0;
        
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,1000, 1e-6) );
        
        
	FileStorage ft("tdatsmall_windo.xml", FileStorage::READ );
	FileStorage fl("tlabsmall_windo.xml", FileStorage::READ );
	
	FileNode n = ft["Data"];
	FileNode l = fl["Labels"];
	
	FileNodeIterator it = n.begin(), it_end = n.end();
	FileNodeIterator it2 = l.begin(), it2_end = l.end(); 
	
	for (; it != it_end, it2 != it2_end; ++it, ++it2)
	{
        
	(*it) >> train_data;// train_data.convertTo(train_data,CV_32F);	// grossly inefficient fix this
	(*it2) >> train_label;
	
        cout << "\n Size of tdata::: "<< train_data.size() << "\n";
        cout << "\n Size of tlabels::: "<< train_label.size() << "\n";
        
        if(t_dat.empty())
        {
            t_dat = Mat::zeros(train_data.size(), train_data.type());
            t_lab = Mat::zeros(train_label.size(), train_label.type());
        }
                
        vconcat(t_dat,train_data,t_dat);
        vconcat(t_lab,train_label,t_lab);           // concatenate all the data (careful might cause a memory leak!!! / buffer overflow)
	}
	
	ft.release();
	fl.release();
	
//         cout << "\n tdata size:: "<< t_dat.rows << "    " << t_dat.cols << "\n";
//         cout << "\n tlab size:: "<< t_lab.rows << "    " << t_lab.cols << "\n";
        
        t_dat = normr(t_dat);
        Mat tt_dat;  pca = preprocess_data_train(t_dat,tt_dat,featu);

//         double a,b;
//         minMaxLoc(tt_dat,&a,&b);
//         tt_dat /= b;
        
//         normalize(t_dat,t_dat,0,1,NORM_MINMAX);
        
        svm->trainAuto( TrainData::create( tt_dat, cv::ml::ROW_SAMPLE, t_lab ), 5,SVM::getDefaultGrid(SVM::C));
        
        double CC = svm->getC();
        double gamma = svm->getGamma();
        
        Ptr<SVM> svm2 = SVM::create();
	svm2->setType(SVM::C_SVC);
	svm2->setKernel(SVM::LINEAR);
	svm2->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,1000, 1e-6) );
        svm2->setC(svm->getC());
//         svm2->setGamma(svm->getGamma());
        
        svm2->train(TrainData::create( tt_dat, cv::ml::ROW_SAMPLE, t_lab )); 
   
        Mat newlabs;
        double ncnt = 0,N=0;
        N = double(t_lab.rows);
        for(int i = 0; i < t_lab.rows; i++)
        {
            float yy = svm2->predict(tt_dat.row(i));
            if(t_lab.at<int>(i) == yy ){ncnt++;}
        }
        tra.push_back((ncnt/N)*100);
        
/*    */    
        
	return svm2;
}

double test_model(Ptr <SVM> svm, PCA pca,vector<Mat> &cnf,vector<float> &pr,vector<float> &re,vector<float> &cl, int featu = 70)
{
Mat test_data;
Mat test_labels;
Mat tdat,tlab;

ofstream h;
h.open("labels.txt",ios::out | ios::ate);
FileStorage ft("test_data_small_windo.xml", FileStorage::READ );
FileStorage fl("test_lab_small_windo.xml", FileStorage::READ );

FileNode n = ft["Data"];
FileNode l = fl["Labels"];

Mat k;
cout << "\nTest Model started!!! \n";
FileNodeIterator it = n.begin(), it_end = n.end();
FileNodeIterator it2 = l.begin(), it2_end = l.end(); 

double ncnt = 0, N = 0;
double ncnt2 = 0, N2 = 0;

for (; it != it_end, it2 != it2_end; ++it, ++it2)
{

(*it) >> tdat;
(*it2) >> tlab;

if(test_data.empty())
{
    test_data = Mat::zeros(tdat.size(),tdat.type());
    test_labels = Mat::zeros(tlab.size(),tlab.type());
    
}



vconcat(test_data,tdat,test_data);
vconcat(test_labels,tlab,test_labels);
}

cout << "\n Data size :: "<< test_data.rows << "     " << test_data.cols <<"\n";
ft.release();
fl.release();

Mat newlabs;

test_data = normr(test_data);
Mat tt;  preprocess_data_test(test_data, pca , tt,featu);


// double a,b;
// minMaxLoc(tt,&a,&b);
// tt /= b;

cout << "\n Test Data size :: "<< tt.rows << "     " << tt.cols <<"\n";
// normalize(test_data,test_data,0,1,NORM_MINMAX);



    for(int i = 0; i < tt.rows; i++)
    {
        float kk = svm->predict(tt.row(i));
        h << "\nindex:::" << i << "pred_label:::" << kk << "act_lab:::" << test_labels.at<int>(i)  ; 

        newlabs.push_back(kk);
        
        if (test_labels.at<int>(i) != 4)
        {
            if(test_labels.at<int>(i) == kk){ncnt2++;}
            N2++;
        }

        if(test_labels.at<int>(i) == kk){ncnt++;}
    }
            N = double(tt.rows);

// cout << "\n SB Test Accuracy     :: " << (ncnt2/N2) * 100 <<"% \n";
// cout << "\n Correct SB Indexes:: "<< ncnt2 << "\n Actual Indexes :: "<< N2 << "\n";
// 
// 
// cout << "\n Classification accuracy is :: " << (ncnt/N) * 100 <<"% \n";
// cout << "\n Correct Test Labels:: "<< ncnt << "\n Total Labels :: "<< N << "\n";

newlabs.convertTo(newlabs,CV_32SC1);
precision_recall(test_labels, newlabs,pr,re);
cl.push_back((ncnt/N)*100);
Mat Y = confusion_mat(test_labels,newlabs);
cnf.push_back(Y);

h.close();

return (ncnt/N)*100;

}

void handle_data()			////// Handle Data/// //// Latest Revision
{
///////////////////////// CAREFUL GENERATE DATA(_forked) IS FORKED!! MISUSING IT MAY CAUSE A FORK BOMB!!!!!!!!!!!!!!!!!!! //////////////////////////////
cout << "\nData Generation Started!! \n";

int dlimiter_on = 1, nframesize = 590;
int i = 0,feat = 19;
int testmode = 0; int win_length = 15;

cout << "\n Data Generation Complete!! \n";
vector<float> pr;
vector<float> re;
vector<float> cl;
vector<float> tra;
vector<int> ii;
vector<Mat> conf;


PCA pca;
Ptr <SVM> svm = train_svm(pca,tra,feat,0);  // dlimiter is on with normal frame limit of 330 frames per training video

VideoCapture cap;
cap.open("/home/user/RPI/PATTERN RECOGNITION/Project 1/Test Data/Sagaofth1957.mp4"); // work on automating reading this file use vector strings

if(!cap.isOpened())
{
    cout << "\n ERROR IN OPENING VIDEO FILE!! SKIPPING VIDEO FILE!! \n";
    return ;
}
int iii = 0;
while( cap.get(CV_CAP_PROP_POS_FRAMES) < cap.get(CV_CAP_PROP_FRAME_COUNT) )
{
    Mat tt = normr(getcompoundfeat(get_window(cap,i, cap.get(CV_CAP_PROP_FRAME_COUNT) - 1,win_length),testmode));
    Mat t;
    preprocess_data_test( tt , pca ,t,feat);
    
int y = int (svm->predict( t ));


if(y != 4)
{
    iii = 0;          // SB encountered flag reset  
}

if(y == 4 && iii == 0)
{
    cout << "\n Frame found::: \n";
    Mat fram;
    cap.set(CV_CAP_PROP_POS_FRAMES,i);
    cap >> fram;
    
    string imgpath("/home/user/RPI/PATTERN RECOGNITION/Project 1/Extracted_Frames/");
    stringstream ii; 
    ii<<i;
    imgpath.append("Nfram");
    imgpath.append(ii.str());
    imgpath.append(".jpeg");
    int kk = imwrite(imgpath.data(),fram);
    iii = 1; // flag set, ensures that only first frame of found shot returned
}


i++;
}


double c_acc = test_model(svm,pca,conf,pr,re,cl,11);

}


int main()
{
ff.open("feat_values.txt", ios::out);
handle_data();
ff.close();
return 1;

}
/*
int main99()
{
Mat test_data, train_data, test_labels, train_labels;
vector<float> test_labs;
int n = 0;

process_video("/home/user/RPI/PATTERN RECOGNITION/Project 1/Training Data/Adelante1959_2.mp4",train_data,train_labels,-1,9000);
process_video("/home/user/RPI/PATTERN RECOGNITION/Project 1/Training Data/Adelante1959_2.mp4",test_data,test_labels,9001,12900);

cout <<"\n Data Obtained !!!\n";

cout <<"\n Training Started !!!\n";
Ptr <SVM> svm = get_SVM_model(train_data,train_labels);
cout <<"\n Training Done !!!\n";
svm->save("9000_model_test_data.svm");

FileStorage file_p("train_data.txt", FileStorage::WRITE);
file_p << train_data;
file_p.release();

FileStorage file_p2("train_labels.txt", FileStorage::WRITE);
file_p2 << train_labels;
file_p2.release();

FileStorage file_p3("test_data.txt", FileStorage::WRITE);
file_p2 << test_data;
file_p2.release();

FileStorage file_p4("test_labels.txt", FileStorage::WRITE);
file_p4 << test_labels;
file_p4.release();


for(int i = 0; i< test_data.rows; i++)
{
	if( int(svm->predict(test_data.row(i))) == test_labels.at<int>(i) ) {n++;}
}

cout <<"\n Classifier Accuracy:: "<< (n/(test_data.rows) )* 100 <<"\n";

return 1;
}


*/
