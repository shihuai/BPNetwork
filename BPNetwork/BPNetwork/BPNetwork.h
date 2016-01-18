#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <vector>
#include <cmath>

using namespace std;

typedef struct NetParam{
	double		neda;
	int			sampleNum;
	int			nInputNodes;
	int			nOutPutNodes;
	int			nHideLayers;
	vector<int> nHideLayerNodes;
}BPNetParam;						//BP神经网络结构的参数

typedef struct weight{
	double *W;
}Weight;							//BP神经网络中的神经元节点之间的权值

typedef struct outputNode{
	double *data;
}OutputNode;						//BP网络中每个神经元节点的输出值

class BPNetwork{

public:
	BPNetwork();
	BPNetwork(BPNetParam &, vector<vector<double> > &, vector<vector<double> > &);
	void initialInput(vector<vector<double> > &);
	void initialOutput(vector<vector<double> > &);
	void initialWeight();
	void constructNetork();
	void buildInputNodes();
	void buildHideLayerNodes();
	void buildOutputLayerNodes();
	void buildHideToHideWeight();
	void buildHideToOutputWeight();
	void clearHideLayerOutputNodes();
	void clearHideToHideWeight();
	void clearHideToOutputWeight();
	void calculateOutput();
	void calculateHideToHideOutput(int , int);
	void calculateHideToOutput(int );
	void calculateDeltaWeight();
	void adjustBPWeight();
	void adjustOutputLayerWeight();
	void adjustHideLayerWeight(int);
	void train(int , double);
	void save(string);
	void predict(vector<vector<double> > &);
	void load(string);
	~BPNetwork();

private:
	double limitValue_0_1(double);
	double getError();

private:
	BPNetParam					bpNetParam;
	vector<vector<double> >		samples;
	vector<vector<double> >		expectOuput;
	vector<vector<double> >		realOutput;
	vector<vector<double> >		hideToOutputDelta;
	vector<Weight>				hideToOutputWeight;
	vector<vector<OutputNode> > hideLayerOutput;
	vector<vector<OutputNode> > hideLayerOutputDelta;
	vector<vector<Weight> >		hideLayerWeight;
};