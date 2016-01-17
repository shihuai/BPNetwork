#include "BPNetwork.h"
#include <cstdlib>
#include <time.h>

BPNetwork::BPNetwork(BPNetParam &param, vector<vector<double> > &inputSample, vector<vector<double> > &outputValue)
{
	bpNetParam.sampleNum	= param.sampleNum;
	bpNetParam.nInputNodes	= param.nInputNodes;
	bpNetParam.nOutPutNodes = param.nOutPutNodes;
	bpNetParam.nHideLayers	= param.nHideLayers;
	
	for(int i = 0; i < bpNetParam.nHideLayers; ++ i)
	{
		bpNetParam.nHideLayerNodes.push_back(param.nHideLayerNodes[i]);
	}

	constructNetork();
	initialInput(inputSample);
	initialOutput(outputValue);
	initialWeight();
}

void BPNetwork::constructNetork()
{
	//***************构建输入节点**********************//
	for(int i = 0; i < bpNetParam.sampleNum; ++ i)
	{
		vector<double> tempSampleInput(bpNetParam.nInputNodes + 1);

		samples.push_back(tempSampleInput);
	}

	//**************构建隐藏层节点的输出节点********************//
	for(int i = 0; i < bpNetParam.nHideLayers; ++ i)
	{
		vector<OutputNode> tempOutputNode(bpNetParam.sampleNum);

		for(int j = 0; j < bpNetParam.sampleNum; ++ j)
		{
			tempOutputNode[j].data = new double[bpNetParam.nHideLayerNodes[i]];
		}

		hideLayerOutput.push_back(tempOutputNode);
	}

	//**************构建输出层的输出节点************************//
	for(int i = 0; i < bpNetParam.sampleNum; ++ i)
	{
		vector<double> tempRealOutput(bpNetParam.nOutPutNodes);
		vector<double> tempExpectOutput(bpNetParam.nOutPutNodes);
		
		realOutput.push_back(tempRealOutput);
		expectOuput.push_back(tempExpectOutput);
	}

	//*************构建输入层一直到最后隐层的权值链接********//
	for(int i = 0; i < bpNetParam.nHideLayers; ++ i)
	{
		vector<Weight> tempHideLayerWeight(bpNetParam.nHideLayerNodes[i] + 1);
		vector<Weight> tempHideLayerWTemp(bpNetParam.nHideLayerNodes[i] + 1);
		vector<Weight> tempHideLayerDelta(bpNetParam.nHideLayerNodes[i] + 1);

		if(i == 0)
		{
			for(int j = 0; j < bpNetParam.nHideLayerNodes[i]; ++ j)
			{
				tempHideLayerWeight[j].W	= new double[bpNetParam.nInputNodes + 1];
				tempHideLayerWTemp[j].W		= new double[bpNetParam.nInputNodes + 1];
				tempHideLayerDelta[j].W		= new double[bpNetParam.nInputNodes + 1];
			}                                                                                                    
		}
		else
		{
			for(int j = 0; j < bpNetParam.nHideLayerNodes[i]; ++ j)
			{
				tempHideLayerWeight[j].W	= new double[bpNetParam.nHideLayerNodes[i - 1] + 1];
				tempHideLayerWTemp[j].W		= new double[bpNetParam.nHideLayerNodes[i - 1] + 1];
				tempHideLayerDelta[j].W		= new double[bpNetParam.nHideLayerNodes[i - 1] + 1];
			}
		}

		hideLayerWeight.push_back(tempHideLayerWeight);
		hideLayerTempWeight.push_back(tempHideLayerWTemp);
		hideLayerDelta.push_back(tempHideLayerDelta);
	}

	//******************构建最后隐层到输出层的权值链接*********//
	for(int i = 0; i < bpNetParam.nOutPutNodes; ++ i)
	{
		Weight tempHideToOutputWeight;
		Weight tempHideToOutputTempWeight;
		Weight tempHideToOutputDelta;
		
		tempHideToOutputWeight.W		= new double[bpNetParam.nHideLayerNodes[bpNetParam.nHideLayers - 1] + 1];
		tempHideToOutputTempWeight.W	= new double[bpNetParam.nHideLayerNodes[bpNetParam.nHideLayers - 1] + 1];
		tempHideToOutputDelta.W			= new double[bpNetParam.nHideLayerNodes[bpNetParam.nHideLayers - 1] + 1];

		hideToOutputWeight.push_back(tempHideToOutputWeight);
		hideToOutputLayerTempWeight.push_back(tempHideToOutputTempWeight);
		hideToOutputLayerDeltaWeight.push_back(tempHideToOutputDelta);
	}

}

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
					hideLayerWeight[i][j].W[k]		= rand() % 100 * 0.01;
					hideLayerTempWeight[i][j].W[k]	= 0.0;
					hideLayerDelta[i][j].W[k]		= 0.0;
				}
			}
			else
			{
				for(int k = 0; k <= bpNetParam.nHideLayerNodes[i - 1]; ++ k)
				{
					hideLayerWeight[i][j].W[k]		= rand() % 100 * 0.01;
					hideLayerTempWeight[i][j].W[k]	= 0.0;
					hideLayerDelta[i][j].W[k]		= 0.0;
				}
			}
		}
	}

	for(int i = 0; i < bpNetParam.nOutPutNodes; ++ i)
	{
		for(int j = 0; j <= bpNetParam.nHideLayerNodes[bpNetParam.nHideLayers - 1]; ++ j)
		{
			hideToOutputWeight[i].W[j]				= rand() % 100 * 0.01;
			hideToOutputLayerTempWeight[i].W[j]		= 0.0;
			hideToOutputLayerDeltaWeight[i].W[j]	= 0.0;
		}
	}
}

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
		}
	}

}

void BPNetwork::calculateHideToOutput(int n)
{
	for(int j = 0; j < bpNetParam.nOutPutNodes; ++ j)
	{
		realOutput[n][j] = hideToOutputWeight[j].W[bpNetParam.nHideLayerNodes[bpNetParam.nHideLayers - 1]];

		for(int k = 0; k < bpNetParam.nHideLayerNodes[bpNetParam.nHideLayers - 1]; ++ k)
		{
			realOutput[n][j] += (hideToOutputWeight[j].W[k] * hideLayerOutput[bpNetParam.nHideLayers - 1][n].data[k]);
		}
	}
}

double BPNetwork::limitValue_0_1(double value)
{
	if(value >= 0.9999)
		value = 0.9999;

	if(value < 0.0001)
		value = 0.0001;

	return value;
}

void BPNetwork::calculateOutput()
{
	for(int n = 0; n < bpNetParam.sampleNum; ++ n)
	{
		for(int i = 0; i < bpNetParam.nHideLayers; ++ i)
		{
			calculateHideToHideOutput(n, i);
		}

		calculateHideToOutput(n);

		for(int j = 0; j < bpNetParam.nOutPutNodes; ++ j)
		{
			realOutput[n][j] = limitValue_0_1(realOutput[n][j]);

			for(int k = 0; k < bpNetParam.nHideLayerNodes[bpNetParam.nHideLayers - 1]; ++ k)
			{
				hideToOutputLayerDeltaWeight[j].W[k] = (expectOuput[n][j] - realOutput[n][j]) * realOutput[n][j] * (1 - realOutput[n][j]);
			}
		}

		for(int i = bpNetParam.nHideLayers - 1; i >= 0; -- i)
		{
			for(int j = 0; j < bpNetParam.nHideLayerNodes[i]; ++ j)
			{
				hideLayerOutput[i][n].data[j] = limitValue_0_1(hideLayerOutput[i][n].data[j]);

				//for(int k = 0; k < bpNetParam)
			}
		}
	}
}

void BPNetwork::train(int iteration, double errorLevel)
{
	int count = 0;
	double newError = 0.0;
	double oldError = 0.0;

	while(count < iteration)
	{
		calculateOutput();

		calculateDelta();

		adjustBPWeight();
	}
}