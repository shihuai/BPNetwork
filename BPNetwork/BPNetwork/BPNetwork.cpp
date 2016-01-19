#include "BPNetwork.h"
#include <cstdlib>
#include <time.h>

BPNetwork::BPNetwork()
{}

BPNetwork::BPNetwork(BPNetParam &param, vector<vector<double> > 
					&inputSample, vector<vector<double> > &outputValue)
{
	bpNetParam.sampleNum	= param.sampleNum;
	bpNetParam.nInputNodes	= param.nInputNodes;
	bpNetParam.nOutPutNodes = param.nOutPutNodes;
	bpNetParam.nHideLayers	= param.nHideLayers;
	bpNetParam.neda			= param.neda;
	
	for(int i = 0; i < bpNetParam.nHideLayers; ++ i)
	{
		bpNetParam.nHideLayerNodes.push_back(param.nHideLayerNodes[i]);
	}

	constructNetork();
	initialInput(inputSample);
	initialOutput(outputValue);
	initialWeight();
}
//***************��������ڵ�******************************//
void BPNetwork::buildInputNodes()
{
	for(int i = 0; i < bpNetParam.sampleNum; ++ i)
	{
		vector<double> tempSampleInput(bpNetParam.nInputNodes + 1);

		samples.push_back(tempSampleInput);
	}
}
//**************�������ز�ڵ������ڵ�********************//
void BPNetwork::buildHideLayerNodes()
{
	for(int i = 0; i < bpNetParam.nHideLayers; ++ i)
	{
		vector<OutputNode> tempOutputNode(bpNetParam.sampleNum);
		vector<OutputNode> tempHideLayerDelta(bpNetParam.sampleNum);

		for(int j = 0; j < bpNetParam.sampleNum; ++ j)
		{
			tempOutputNode[j].data		= new double[bpNetParam.nHideLayerNodes[i]];
			tempHideLayerDelta[j].data	= new double[bpNetParam.nHideLayerNodes[i]];
		}

		hideLayerOutput.push_back(tempOutputNode);
		hideLayerOutputDelta.push_back(tempHideLayerDelta);
	}
}
//**************��������������ڵ�************************//
void BPNetwork::buildOutputLayerNodes()
{
	for(int i = 0; i < bpNetParam.sampleNum; ++ i)
	{
		vector<double> tempRealOutput(bpNetParam.nOutPutNodes);
		vector<double> tempExpectOutput(bpNetParam.nOutPutNodes);
		vector<double> tempOutputLayerDelta(bpNetParam.nOutPutNodes);
		
		realOutput.push_back(tempRealOutput);
		expectOuput.push_back(tempExpectOutput);
		hideToOutputDelta.push_back(tempOutputLayerDelta);
	}
}
//*************���������һֱ����������Ȩֵ����************//
void BPNetwork::buildHideToHideWeight()
{
	for(int i = 0; i < bpNetParam.nHideLayers; ++ i)
	{
		vector<Weight> tempHideLayerWeight(bpNetParam.nHideLayerNodes[i]);

		if(i == 0)
		{
			for(int j = 0; j < bpNetParam.nHideLayerNodes[i]; ++ j)
			{
				tempHideLayerWeight[j].W = new double[bpNetParam.nInputNodes + 1];
			}                                                                                                    
		}
		else
		{
			for(int j = 0; j < bpNetParam.nHideLayerNodes[i]; ++ j)
			{
				tempHideLayerWeight[j].W = new double[bpNetParam.nHideLayerNodes[i - 1] + 1];
			}
		}

		hideLayerWeight.push_back(tempHideLayerWeight);
	}
}
//******************����������㵽������Ȩֵ����***********//
void BPNetwork::buildHideToOutputWeight()
{
	for(int i = 0; i < bpNetParam.nOutPutNodes; ++ i)
	{
		Weight tempHideToOutputWeight;
		
		tempHideToOutputWeight.W = new double[bpNetParam.nHideLayerNodes[bpNetParam.nHideLayers - 1] + 1];

		hideToOutputWeight.push_back(tempHideToOutputWeight);
	}
}
//******************��ʼ����BP����*************************//
void BPNetwork::constructNetork()
{
	buildInputNodes();
	buildHideLayerNodes();
	buildOutputLayerNodes();
	buildHideToHideWeight();
	buildHideToOutputWeight();
}
//*****************��ʼ������ڵ�**************************//
void BPNetwork::initialInput(vector<vector<double> > &inputSample)
{
	for(int i = 0; i < bpNetParam.sampleNum; ++ i)
	{
		for(int j = 0; j < bpNetParam.nInputNodes; ++ j)
		{
			samples[i][j] = inputSample[i][j];
		}
	}
}
//*****************��ʼ����������ڵ�**********************//
void BPNetwork::initialOutput(vector<vector<double> > &outputValue)
{
	for(int i = 0; i < bpNetParam.sampleNum; ++ i)
	{
		for(int j = 0; j < bpNetParam.nOutPutNodes; ++ j)
		{
			expectOuput[i][j] = outputValue[i][j];
		}
	}
}
//*****************��ʼ�����������Ȩֵ********************//
void BPNetwork::initialWeight()
{
	srand((unsigned)time(NULL));

	for(int i = 0; i < bpNetParam.nHideLayers; ++ i)
	{
		for(int j = 0; j < bpNetParam.nHideLayerNodes[i]; ++ j)
		{
			if(i == 0)
			{
				for(int k = 0; k <= bpNetParam.nInputNodes; ++ k)
				{
					hideLayerWeight[i][j].W[k] = MAXWEIGHT * (0.1-2 * (float)rand() / SCALEWEIGHT);
				}
			}
			else
			{
				for(int k = 0; k <= bpNetParam.nHideLayerNodes[i - 1]; ++ k)
				{
					hideLayerWeight[i][j].W[k] = MAXWEIGHT * (0.1-2 * (float)rand() / SCALEWEIGHT);
				}
			}
		}
	}

	for(int i = 0; i < bpNetParam.nOutPutNodes; ++ i)
	{
		for(int j = 0; j <= bpNetParam.nHideLayerNodes[bpNetParam.nHideLayers - 1]; ++ j)
		{
			hideToOutputWeight[i].W[j] = MAXWEIGHT * (0.1-2 * (float)rand() / SCALEWEIGHT);
		}
	}
}
//*****************�������һ����������******************//
void BPNetwork::calculateHideToHideOutput(int n, int i)
{
	if(i == 0)
	{
		for(int j = 0; j < bpNetParam.nHideLayerNodes[i]; ++ j)
		{
			hideLayerOutput[i][n].data[j] = hideLayerWeight[i][j].W[bpNetParam.nInputNodes];

			for(int k = 0; k < bpNetParam.nInputNodes; ++ k)
			{
				hideLayerOutput[i][n].data[j] += (hideLayerWeight[i][j].W[k] * samples[n][k]);
			}
			hideLayerOutput[i][n].data[j] = 1.0 / (1.0 + exp(-hideLayerOutput[i][n].data[j]));
		}
	}
	else
	{
		for(int j = 0; j < bpNetParam.nHideLayerNodes[i]; ++ j)
		{
			hideLayerOutput[i][n].data[j] = hideLayerWeight[i][j].W[bpNetParam.nHideLayerNodes[i - 1]];

			for(int k = 0; k < bpNetParam.nHideLayerNodes[i - 1]; ++ k)
			{
				hideLayerOutput[i][n].data[j] += (hideLayerWeight[i][j].W[k] * hideLayerOutput[i - 1][n].data[k]);
			}
			hideLayerOutput[i][n].data[j] = 1.0 / (1.0 + exp(-hideLayerOutput[i][n].data[j]));
		}
	}

}
//*****************����������������********************//
void BPNetwork::calculateHideToOutput(int n)
{
	for(int j = 0; j < bpNetParam.nOutPutNodes; ++ j)
	{
		realOutput[n][j] = hideToOutputWeight[j].W[bpNetParam.nHideLayerNodes[bpNetParam.nHideLayers - 1]];

		for(int k = 0; k < bpNetParam.nHideLayerNodes[bpNetParam.nHideLayers - 1]; ++ k)
		{
			realOutput[n][j] += (hideToOutputWeight[j].W[k] * hideLayerOutput[bpNetParam.nHideLayers - 1][n].data[k]);
		}
		realOutput[n][j] = 1.0 / (1.0 + exp(-realOutput[n][j]));
	}
}
//*****************�淶���*******************************//
double BPNetwork::limitValue_0_1(double value)
{
	if(value > 0.9999)
		value = 0.9999;

	if(value < 0.0001)
		value = 0.0001;

	return value;
}
//*****************����ÿһ������************************//
void BPNetwork::calculateOutput()
{
	for(int n = 0; n < bpNetParam.sampleNum; ++ n)
	{
		for(int i = 0; i < bpNetParam.nHideLayers; ++ i)
		{
			calculateHideToHideOutput(n, i);
		}

		calculateHideToOutput(n);
		
		//******************����ÿ��������������Ӧ�ĵ¶���ֵ*************//
		for(int j = 0; j < bpNetParam.nOutPutNodes; ++ j)
		{
			realOutput[n][j] = limitValue_0_1(realOutput[n][j]);

			hideToOutputDelta[n][j] = (expectOuput[n][j] - realOutput[n][j]) * realOutput[n][j] * (1 - realOutput[n][j]);
		}

		//*****************����ÿ��������ÿ�������Ӧ�ĵ¶���ֵ************//
		for(int i = bpNetParam.nHideLayers - 1; i >= 0; -- i)
		{
			for(int j = 0; j < bpNetParam.nHideLayerNodes[i]; ++ j)
			{
				double sum = 0.0;
				hideLayerOutput[i][n].data[j] = limitValue_0_1(hideLayerOutput[i][n].data[j]);

				if(i == bpNetParam.nHideLayers - 1)
				{
					for(int k = 0; k < bpNetParam.nOutPutNodes; ++ k)
					{
						sum += hideToOutputWeight[k].W[j] * hideToOutputDelta[n][k];
					}
				}
				else
				{
					for(int k = 0; k < bpNetParam.nHideLayerNodes[i + 1]; ++ k)
					{
						sum += hideLayerWeight[i + 1][k].W[j] * hideLayerOutputDelta[i + 1][n].data[k];
					}
				}

				hideLayerOutputDelta[i][n].data[j] = sum * hideLayerOutput[i][n].data[j] * (1.0 - hideLayerOutput[i][n].data[j]);
			}
		}
	}
}
//*****************�������������������Ȩֵ***************//
void BPNetwork::adjustOutputLayerWeight()
{
	double dw = 0.0;

	for(int j = 0; j < bpNetParam.nOutPutNodes; ++ j)
	{
		dw = 0.0;
		for(int n = 0; n < bpNetParam.sampleNum; ++ n)
		{
			dw += hideToOutputDelta[n][j];
		}

		dw *=  bpNetParam.neda;
		hideToOutputWeight[j].W[bpNetParam.nHideLayerNodes[bpNetParam.nHideLayers - 1]] += dw;
	
		for(int k = 0; k < bpNetParam.nHideLayerNodes[bpNetParam.nHideLayers - 1]; ++ k)
		{
			dw = 0.0;
			for(int n = 0; n < bpNetParam.sampleNum; ++ n)
			{
				dw += hideToOutputDelta[n][j] * hideLayerOutput[bpNetParam.nHideLayers - 1][n].data[k];
			}

			dw *= bpNetParam.neda;
			hideToOutputWeight[j].W[k] += dw;
		}
	}
}
//*****************��������֮�������Ȩֵ*******************//
void BPNetwork::adjustHideLayerWeight(int i)
{
	double dw = 0.0;

	if(i == 0)
	{
		for(int j = 0; j < bpNetParam.nHideLayerNodes[i]; ++ j)
		{
			dw = 0.0;
			for(int n = 0; n < bpNetParam.sampleNum; ++ n)
			{
				dw += hideLayerOutputDelta[i][n].data[j];
			}
			
			dw *= bpNetParam.neda;
			hideLayerWeight[i][j].W[bpNetParam.nInputNodes] += dw;

			for(int k = 0; k < bpNetParam.nInputNodes; ++ k)
			{
				dw = 0.0;
				for(int n = 0; n < bpNetParam.sampleNum; ++ n)
				{
					dw += (hideLayerOutputDelta[i][n].data[j] * samples[n][k]);
				}

				dw *= bpNetParam.neda;
				hideLayerWeight[i][j].W[k] += dw;
			}
		}
	}
	else
	{
		for(int j = 0; j < bpNetParam.nHideLayerNodes[i]; ++ j)
		{
			dw = 0.0;
			for(int n = 0; n < bpNetParam.sampleNum; ++ n)
			{
				dw += hideLayerOutputDelta[i][n].data[j];
			}

			dw *= bpNetParam.neda;
			hideLayerWeight[i][j].W[bpNetParam.nHideLayerNodes[i - 1]] += dw;

			for(int k = 0; k < bpNetParam.nHideLayerNodes[i - 1]; ++ k)
			{
				dw = 0.0;
				for(int n = 0; n < bpNetParam.sampleNum; ++ n)
				{
					dw += (hideLayerOutputDelta[i][n].data[j] * hideLayerOutput[i - 1][n].data[k]);
				}

				dw *= bpNetParam.neda;
				hideLayerWeight[i][j].W[k] += dw;
			}
		}
	}
}
//****************��������BP�����Ȩֵ*********************//
void BPNetwork::adjustBPWeight()
{
	adjustOutputLayerWeight();

	for(int j = bpNetParam.nHideLayers - 1; j >= 0; --j)
	{
		adjustHideLayerWeight(j);
	}
}
//****************������*********************************//
double BPNetwork::getError()
{
	double error = 0.0;
	for(int n = 0; n < bpNetParam.sampleNum; ++ n)
	{
		for(int j = 0; j < bpNetParam.nOutPutNodes; ++ j)
		{
			error += (realOutput[n][j] - expectOuput[n][j]) * (realOutput[n][j] - expectOuput[n][j]);
		}
	}

	return error;
}
//****************��ʼѵ��*********************************//
void BPNetwork::train(int iteration, double errorLevel)
{
	int		count	 = 0;
	double	newError = 0.0;
	double	oldError = 0.0;

	while(count < iteration)
	{
		calculateOutput();
		adjustBPWeight();

		newError = getError();
		newError /= (bpNetParam.nOutPutNodes * bpNetParam.sampleNum);

		if(newError < errorLevel)
		{
			break;
		}

		++ count;
		//cout << "newError:" << newError << endl;
	}
	cout << count << endl;
	ofstream out("F:\\realOutput.txt", ios::out);

	for(int n = 0; n < bpNetParam.sampleNum; ++ n)
	{
		for(int i = 0; i < bpNetParam.nOutPutNodes; ++ i)
		{
			out << realOutput[n][i] << " ";
		}
		out << endl;
	}
	out.close();
}
//****************Ԥ����������*****************************//
void BPNetwork::predict(vector<vector<double> > &testSample)
{
	double sum			= 0.0;
	int inputSampleNum	= testSample.size();
	vector<OutputNode> hideLayerOutputNodes;
	vector<double> outputLayerNodes(bpNetParam.nOutPutNodes);

	for(int i = 0; i < bpNetParam.nHideLayers; ++ i)
	{
		OutputNode tempOutputNode;
		
		tempOutputNode.data = new double[bpNetParam.nHideLayerNodes[i]];

		hideLayerOutputNodes.push_back(tempOutputNode);
	}

	for(int n = 0; n < inputSampleNum; ++ n)
	{
		for(int i = 0; i < bpNetParam.nHideLayers; ++ i)
		{
			for(int j = 0; j < bpNetParam.nHideLayerNodes[i]; ++ j)
			{
				sum = 0.0;
				if(i == 0)
				{
					sum = hideLayerWeight[i][j].W[bpNetParam.nInputNodes];
					for(int k = 0; k < bpNetParam.nInputNodes; ++ k)
					{
						sum += hideLayerWeight[i][j].W[k] * testSample[n][k];
					}
					hideLayerOutputNodes[i].data[j] = 1 / (1 + exp(-sum));
				}
				else
				{
					sum = hideLayerWeight[i][j].W[bpNetParam.nHideLayerNodes[i - 1]];
					for(int k = 0; k < bpNetParam.nHideLayerNodes[i - 1]; ++ k)
					{
						sum += hideLayerWeight[i][j].W[k] * hideLayerOutputNodes[i - 1].data[k];
					}
					hideLayerOutputNodes[i].data[j] = 1 / (1 + exp(-sum));
				}
			}
		}

		for(int j = 0; j < bpNetParam.nOutPutNodes; ++ j)
		{
			sum = hideToOutputWeight[j].W[bpNetParam.nHideLayerNodes[bpNetParam.nHideLayers - 1]];
			for(int k = 0; k < bpNetParam.nHideLayerNodes[bpNetParam.nHideLayers - 1]; ++ k)
			{
				sum += hideToOutputWeight[j].W[k] * hideLayerOutputNodes[bpNetParam.nHideLayers - 1].data[k];
			}
			outputLayerNodes[j] = 1 / (1 + exp(-sum));
		}

		for(int j = 0; j < bpNetParam.nOutPutNodes; ++ j)
		{
			cout << outputLayerNodes[j] << " ";
		}
		cout << endl;
	}
}
//****************���������籣��***************************//
void BPNetwork::save(string path)
{
	ofstream out(path, ios::out);

	out << bpNetParam.nInputNodes << " " << bpNetParam.nOutPutNodes << " " << bpNetParam.nHideLayers << endl;

	for(int i = 0; i < bpNetParam.nHideLayers; ++ i)
	{
		out << bpNetParam.nHideLayerNodes[i] << " ";
	}

	cout << endl;

	for(int i = 0; i < bpNetParam.nHideLayers; ++ i)
	{
		if(i == 0)
		{
			for(int j = 0; j < bpNetParam.nHideLayerNodes[i]; ++ j)
			{
				for(int k = 0; k <= bpNetParam.nInputNodes; ++ k)
				{
					out << hideLayerWeight[i][j].W[k] << " ";
				}

				out << endl;
			}
		}
		else
		{
			for(int j = 0; j < bpNetParam.nHideLayerNodes[i]; ++ j)
			{
				for(int k = 0; k <= bpNetParam.nHideLayerNodes[i - 1]; ++ k)
				{
					out << hideLayerWeight[i][j].W[k] << " ";
				}

				out << endl;
			}
		}
	}

	for(int j = 0; j < bpNetParam.nOutPutNodes; ++ j)
	{
		for(int k = 0; k <= bpNetParam.nHideLayerNodes[bpNetParam.nHideLayers - 1]; ++ k)
		{
			out << hideToOutputWeight[j].W[k] << " ";
		}

		out << endl;
	}

	out.close();
}
//****************����һ��BP����***************************//
void BPNetwork::load(string path)
{
	ifstream in(path, ios::in);
	
	if(hideLayerOutput.size() != 0)
	{
		clearHideLayerOutputNodes();
	}

	if(hideLayerWeight.size() != 0)
	{
		clearHideToHideWeight();
	}

	if(hideToOutputWeight.size() != 0)
	{
		clearHideToOutputWeight();
	}

	if(bpNetParam.nHideLayerNodes.size() != 0)
	{
		bpNetParam.nHideLayerNodes.clear();
	}

	in >> bpNetParam.nInputNodes >> bpNetParam.nOutPutNodes >> bpNetParam.nHideLayers;

	int temp;
	for(int i = 0; i < bpNetParam.nHideLayers; ++ i)
	{
		in >> temp;

		bpNetParam.nHideLayerNodes.push_back(temp);
	}

	buildHideToHideWeight();
	buildHideToOutputWeight();

	for(int i = 0; i < bpNetParam.nHideLayers; ++ i)
	{
		for(int j = 0; j < bpNetParam.nHideLayerNodes[i]; ++ j)
		{
			if(i == 0)
			{
				for(int k = 0; k <= bpNetParam.nInputNodes; ++ k)
				{
					in >> hideLayerWeight[i][j].W[k];
				}
			}
			else
			{
				for(int k = 0; k <= bpNetParam.nHideLayerNodes[i - 1]; ++ k)
				{
					in >> hideLayerWeight[i][j].W[k];
				}
			}
		}
	}

	for(int j = 0; j < bpNetParam.nOutPutNodes; ++ j)
	{
		for(int k = 0; k <= bpNetParam.nHideLayerNodes[bpNetParam.nHideLayers - 1]; ++ k)
		{
			in >> hideToOutputWeight[j].W[k];
		}
	}

	in.close();
}
//****************�������������ڵ�**********************//
void BPNetwork::clearHideLayerOutputNodes()
{
	for(int i = 0; i < bpNetParam.nHideLayers; ++ i)
	{
		for(int j = 0; j < bpNetParam.sampleNum; ++ j)
		{
			delete [] hideLayerOutput[i][j].data;
			delete [] hideLayerOutputDelta[i][j].data;
		}
	}
	hideLayerOutputDelta.clear();
	hideLayerOutput.clear();
}
//****************�������������֮�������Ȩֵ***************//
void BPNetwork::clearHideToHideWeight()
{
	for(int i = 0; i < bpNetParam.nHideLayers; ++ i)
	{
		for(int j = 0; j < bpNetParam.nHideLayerNodes[i]; ++ j)
		{
			delete [] hideLayerWeight[i][j].W;
		}
	}

	hideLayerWeight.clear();
}
//****************������㵽�����������Ȩֵ***************//
void BPNetwork::clearHideToOutputWeight()
{
	for(int i = 0; i < bpNetParam.nOutPutNodes; ++ i)
	{
		delete [] hideToOutputWeight[i].W;
	}

	hideToOutputWeight.clear();
}

BPNetwork::~BPNetwork()
{
	if(hideLayerOutput.size() != 0)
	{
		clearHideLayerOutputNodes();
	}

	if(hideLayerWeight.size() != 0)
	{
		clearHideToHideWeight();
	}

	if(hideToOutputWeight.size() != 0)
	{
		clearHideToOutputWeight();
	}
}