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
	BPNetwork(BPNetParam &param, vector<vector<double> > &inputSample, vector<vector<double> > &outputValue);
	void initialInput(vector<vector<double> > &inputSample);
	void initialOutput(vector<vector<double> > &outputValue);
	void initialWeight();
	void constructNetork();
	void calculateOutput();
	void calculateDelta();
	void adjustBPWeight();
	void train(int , double);

private:
	BPNetParam bpNetParam;
	//double 
	vector<vector<double> > samples;
	vector<vector<double> > expectOuput;
	vector<vector<double> > realOutput;
	vector<vector<OutputNode> > hideLayerOutput;
	vector<vector<double> > outputLayerDelta;
	vector<vector<Node> > hideLayerDelta;
	vector<vector<Node> > outputLayerTempWeight;
	vector<vector<Node> > hideLayerTempWeight;
	vector<vector<Weight> > hideLayerWeight;
	vector<Weight> hideToOutputWeight;
	//vector<vector<>>
};