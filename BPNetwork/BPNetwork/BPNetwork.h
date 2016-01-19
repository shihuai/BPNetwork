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
	double		neda;				//ѧϰ����
	int			sampleNum;			//ѵ����������
	int			nInputNodes;		//����ڵ�����
	int			nOutPutNodes;		//����ڵ�����
	int			nHideLayers;		//�������
	vector<int> nHideLayerNodes;	//ÿ������Ľڵ���
}BPNetParam;						//BP������ṹ�Ĳ���

typedef struct weight{
	double *W;
}Weight;							//BP�������е���Ԫ�ڵ�֮���Ȩֵ

typedef struct outputNode{
	double *data;
}OutputNode;						//BP������ÿ����Ԫ�ڵ�����ֵ

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
	BPNetParam					bpNetParam;					//BP����ĸ��ֲ���
	vector<vector<double> >		samples;					//ѵ��������
	vector<vector<double> >		expectOuput;				//�������
	vector<vector<double> >		realOutput;					//ʵ����������
	vector<vector<double> >		hideToOutputDelta;			//��������ĵ¶���ֵ
	vector<Weight>				hideToOutputWeight;			//������㵽����������Ȩֵ
	vector<vector<OutputNode> > hideLayerOutput;			//��������
	vector<vector<OutputNode> > hideLayerOutputDelta;		//����ĵ¶���ֵ
	vector<vector<Weight> >		hideLayerWeight;			//����֮�������Ȩֵ
};