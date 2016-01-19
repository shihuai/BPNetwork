#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <vector>
#include <cmath>

using namespace std;

const double MAXWEIGHT		= ((float)0.3);
const double SCALEWEIGHT	= ((float)32767);

typedef struct NetParam{
	double		neda;				//学习速率
	int			sampleNum;			//训练样本数量
	int			nInputNodes;		//输入节点数量
	int			nOutPutNodes;		//输出节点数量
	int			nHideLayers;		//隐层层数
	vector<int> nHideLayerNodes;	//每层隐层的节点数
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
	BPNetParam					bpNetParam;					//BP网络的各种参数
	vector<vector<double> >		samples;					//训练样本集
	vector<vector<double> >		expectOuput;				//期望输出
	vector<vector<double> >		realOutput;					//实际输出层输出
	vector<vector<double> >		hideToOutputDelta;			//最后输出层的德尔塔值
	vector<Weight>				hideToOutputWeight;			//最后隐层到输出层的链接权值
	vector<vector<OutputNode> > hideLayerOutput;			//隐层的输出
	vector<vector<OutputNode> > hideLayerOutputDelta;		//隐层的德尔塔值
	vector<vector<Weight> >		hideLayerWeight;			//隐层之间的链接权值
};