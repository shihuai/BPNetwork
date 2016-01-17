#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <vector>

using namespace std;

typedef struct NetParam{
	int sampleNum;
	int nInputNodes;
	int nOutPutNodes;
	int nHideLayers;
	vector<int> nHideLayerNodes;
}BPNetParam;			//BP������ṹ�Ĳ���

typedef struct weight{
	double *W;
}Weight;				//BP�������е���Ԫ�ڵ�֮���Ȩֵ

typedef struct outputNode{
	double *data;
}OutputNode;					//BP������ÿ����Ԫ�ڵ�����ֵ

class BPNetwork{

public:
	BPNetwork(BPNetParam &, vector<vector<double> > &, vector<vector<double> > &);
	void initialInput(vector<vector<double> > &);
	void initialOutput(vector<vector<double> > &);
	void initialWeight();
	void constructNetork();
	void calculateOutput();
	void calculateHideToHideOutput(int , int);
	void calculateHideToOutput(int );
	void calculateDelta();
	void adjustBPWeight();
	void train(int , double);

private:
	double limitValue_0_1(double);

private:
	BPNetParam					bpNetParam;
	vector<vector<double> >		samples;
	vector<vector<double> >		expectOuput;
	vector<vector<double> >		realOutput;
	vector<Weight>				hideToOutputLayerDeltaWeight;
	vector<Weight>				hideToOutputWeight;
	vector<Weight>				hideToOutputLayerTempWeight;
	vector<vector<double> >		hideToOutputDelta;
	vector<vector<OutputNode> > hideLayerOutput;
	vector<vector<OutputNode> > hideLayerOutputDelta;
	vector<vector<Weight> >		hideLayerTempWeight;
	vector<vector<Weight> >		hideLayerDelta;
	vector<vector<Weight> >		hideLayerWeight;
};