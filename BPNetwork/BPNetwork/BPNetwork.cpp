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
		vector<double> tempDelta(bpNetParam.nOutPutNodes);
		
		realOutput.push_back(tempRealOutput);
		expectOuput.push_back(tempExpectOutput);
		outputLayerDelta.push_back(tempDelta);
	}

	//*************构建输入层一直到最后隐层的权值链接********//
	for(int i = 0; i < bpNetParam.nHideLayers; ++ i)
	{
		vector<Weight> tempHideLayerWeight(bpNetParam.nHideLayerNodes[i] + 1);

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

	//******************构建最后隐层到输出层的权值链接*********//
	for(int i = 0; i < bpNetParam.nOutPutNodes; ++ i)
	{
		Weight tempHideToOutputWeight;
		
		tempHideToOutputWeight.W = new double[bpNetParam.nHideLayerNodes[bpNetParam.nHideLayers - 1] + 1];

		hideToOutputWeight.push_back(tempHideToOutputWeight);
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
					hideLayerWeight[i][j].W[k] = rand() % 100 * 0.01;
				}
			}
			else
			{
				for(int k = 0; k <= bpNetParam.nHideLayerNodes[i - 1]; ++ k)
				{
					hideLayerWeight[i][j].W[k] = rand() % 100 * 0.01;
				}
			}
		}
	}

	for(int i = 0; i < bpNetParam.nOutPutNodes; ++ i)
	{
		for(int j = 0; j < bpNetParam.nHideLayerNodes[bpNetParam.nHideLayers - 1]; ++ j)
		{
			hideToOutputWeight[i].W[j] = rand() % 100 * 0.01;
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