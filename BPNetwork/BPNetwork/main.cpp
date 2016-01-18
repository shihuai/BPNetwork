#include "BPNetwork.h"

int main()
{
#if 1
	BPNetParam param;
	int temp;
	ifstream in("F:\\Github\\BPNetwork\\BPNetwork\\input.txt", ios::in);

	in >> param.sampleNum >> param.nInputNodes >> param.nOutPutNodes >> param.nHideLayers >> param.neda;

	for(int i = 0; i < param.nHideLayers; ++ i)
	{
		in >> temp;

		param.nHideLayerNodes.push_back(temp);
	}

	vector<vector<double> > inputSample;
	vector<vector<double> > expectOutput;

	for(int i = 0; i < param.sampleNum; ++ i)
	{
		vector<double> tempInputSample(param.nInputNodes);

		for(int j = 0; j < param.nInputNodes; ++ j)
		{
			in >> tempInputSample[j];
		}

		inputSample.push_back(tempInputSample);
	}

	for(int i = 0; i < param.sampleNum; ++ i)
	{
		vector<double> tempExpectOutput(param.nOutPutNodes);

		for(int j = 0; j < param.nOutPutNodes; ++ j)
		{
			in >> tempExpectOutput[j];
		}

		expectOutput.push_back(tempExpectOutput);
	}

	in.close();

	BPNetwork bpNetwork(param, inputSample, expectOutput);

	bpNetwork.train(500, 0);
	
	bpNetwork.save("F:\\BPNetwork.txt");
#else

	BPNetwork bpNetwork;

	bpNetwork.load("F:\\BPNetwork.txt");

#endif

	ifstream inTest("F:\\Github\\BPNetwork\\BPNetwork\\ftest.txt", ios::in);
	vector<vector<double> > testData;
	vector<double> vecTemp(6);
	double dTemp;
	int count = 0;

	while(inTest >> dTemp)
	{
		if(count == 6)
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