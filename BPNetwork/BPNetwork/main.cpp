#include "BPNetwork.h"

int main()
{
	BPNetParam param;
	int temp;

	cin >> param.sampleNum >> param.nInputNodes >> param.nOutPutNodes >> param.nHideLayers >> param.neda;

	for(int i = 0; i < param.nHideLayers; ++ i)
	{
		cin >> temp;

		param.nHideLayerNodes.push_back(temp);
	}

	ifstream in("input.txt", ios::in);
	vector<vector<double> > inputSample;
	vector<vector<double> > expectOutput;

	for(int i = 0; i < param.sampleNum; ++ i)
	{
		vector<double> tempInputSample(param.nInputNodes);

		for(int j = 0; j < param.nInputNodes; ++ j)
		{
			cin >> tempInputSample[j];
		}

		inputSample.push_back(tempInputSample);
	}

	for(int i = 0; i < param.sampleNum; ++ i)
	{
		vector<double> tempExpectOutput(param.nOutPutNodes);

		for(int j = 0; j < param.nOutPutNodes; ++ j)
		{
			cin >> tempExpectOutput[j];
		}

		expectOutput.push_back(tempExpectOutput);
	}

	in.close();

	BPNetwork bpNetwork(param, inputSample, expectOutput);

	bpNetwork.train(50, 0.001);

	ifstream inTest("ftest.txt", ios::in);
	vector<vector<double> > testData;
	vector<double> vecTemp(param.nInputNodes);
	double dTemp;
	int count = 0;

	while(inTest >> dTemp)
	{
		if(count == param.nInputNodes)
		{
			testData.push_back(vecTemp);

			count = 0;
			vecTemp[count ++] = dTemp;
		}

		vecTemp[count ++] = dTemp;
	}

	inTest.close();

	bpNetwork.predict(testData);

	return 0;
}