
#include "shob.hpp"

/// To actually use this you would need to browse down into the code and create some directories and actually require the labels that I generated
/// If it is necessary to use this code and see how this works please contact me at dassg@rpi.edu

/// You'll need
/// a.) The required libraries
/// b.) Pseudo install of running environment (Folders with data which are needed)
/// c.) Actual Training Labels I generated from TRECVID's database along with videos


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
        
        // creating svm instances all the time since my dataset very small. Would be better to save it, but for this project I think this is fine
        
        
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
//         svm->setDegree(3);
        svm->trainAuto( TrainData::create( tt_dat, cv::ml::ROW_SAMPLE, t_lab ), 5,SVM::getDefaultGrid(SVM::C));//,SVM::getDefaultGrid(SVM::GAMMA) );

//         cout <<"\n Gamma:: " << svm->getGamma() <<
        
        Ptr<SVM> svm2 = SVM::create();
	svm2->setType(SVM::C_SVC);
	svm2->setKernel(SVM::LINEAR);
	svm2->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,1000, 1e-6) );
        svm2->setC(svm->getC());
        
//         svm2->setGamma(svm->getGamma());
//         svm2->setCoef0(svm->getCoef0());
//         svm2->setDegree(svm->getDegree());
        
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
        
/*      Ptr<SVM> svm2 = SVM::create();
	svm2->setType(SVM::C_SVC);
	svm2->setKernel(SVM::LINEAR);
	svm2->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,1000, 1e-6) );
        svm2->setC(svm->getC());
        
        svm2->train(TrainData::create( tt_dat, cv::ml::ROW_SAMPLE, t_lab ));    */    
        
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

cout << "\n Data size :: "<< tdat.rows << "     " << tdat.cols <<"\n";

vconcat(test_data,tdat,test_data);
vconcat(test_labels,tlab,test_labels);
}

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

void handle_data()			////// Handle Data
{
cout << "\nData Generation Started!! \n";

int dlimiter_on = 1, nframesize = 590;

// generate_data(string("/home/hulio/RPI/PATTERN RECOGNITION/Project 1/Training Data"),string("/home/hulio/RPI/PATTERN RECOGNITION/Project 1/Training Labels"),string("tdatsmall_windo.xml"),string("tlabsmall_windo.xml"),dlimiter_on,nframesize);         

// This is commented now but you can create your own training set if you replace "tlabsmall_windo" and "tdatsmall_windo" which are labels and data files respectively
// This will generate the data for you, run it once to generate and then from there onwards you can omit it.

cout << "\n Training Data Generated!! \n";

// generate_data(string("/home/hulio/RPI/PATTERN RECOGNITION/Project 1/Test Data"),string("/home/hulio/RPI/PATTERN RECOGNITION/Project 1/Test Labels"),string("test_data_small_windo.xml"),string("test_lab_small_windo.xml"),dlimiter_on,nframesize);         



cout << "\n Data Generation Complete!! \n";
vector<float> pr;
vector<float> re;
vector<float> cl;
vector<float> tra;
vector<int> ii;
vector<Mat> conf;

for (int i = 1; i <=100;i+=2 )
{
PCA pca;
Ptr <SVM> svm = train_svm(pca,tra,i,0);  // dlimiter is on with normal frame limit of 330 frames per training video
double c_acc = test_model(svm,pca,conf,pr,re,cl,i);
ii.push_back(i);
cout <<"\n Processing i " << i <<"\n";    
}


cout << "\n Precision \n";

disp_vectf(pr);

cout << "\n Recall \n"; 

disp_vectf(re);

cout << "\n Classification \n";

disp_vectf(cl);

cout << "\n Features Considered \n"; 

disp_vecti(ii);

cout << "\n Confusion Matrix \n"; 

disp_vectm(conf);

cout << "\n Training Acc \n"; 

disp_vectf(tra);

}

int main()
{
ff.open("feat_values.txt", ios::out);
handle_data();
ff.close();
return 1;

}
