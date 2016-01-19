#include "BPNetwork.h"

int main()
{
	int temp;
#if 0
	BPNetParam param;
	ifstream in("F:\\input.txt", ios::in);

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

	bpNetwork.train(2000, 1.0e-10);

	bpNetwork.save("F:\\BPNetwork.txt");
	
#else	
	BPNetwork bpNetwork;

	bpNetwork.load("F:\\BPNetwork.txt");

#endif

	ifstream inTest("F:\\ftest.txt", ios::in);
	vector<vector<double> > testData;
	vector<double> vecTemp(6);
	double dTemp;
	int count = 0;
	
	inTest >> temp;

	while(inTest >> dTemp)
	{
		if(count == 6)
		{
			testData.push_back(vecTemp);

			count = 0;
			vecTemp[count ++] = dTemp;

			continue;
		}

		vecTemp[count ++] = dTemp;
	}

	inTest.close();

	bpNetwork.predict(testData);

	return 0;
}