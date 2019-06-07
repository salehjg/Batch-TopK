#include <iostream>
#include <string>
#include <stdlib.h>
#include <chrono>
using namespace std;
using namespace std::chrono;

#define CONFIG_LEN (1024)
#define CONFIG_K   (20)

float RandomFloat(float min, float max){
    float scale = rand() / (float) RAND_MAX; /* [0, 1.0] */
    return min + scale * ( max - min );      /* [min, max] */
}

void InitTensor(float *tensor, int len, int mode){
	if(mode==0){
		for(int i=0;i<len;i++){
			tensor[i] = RandomFloat(-2.0f,2.0f);
		}
	}
}

void CloneTensor(const float *srcTn, float *dstTn, int len){
	for(int i=0;i<len;i++){
		dstTn[i]=srcTn[i];
	}
}

void SelectionSortAscending(float* tensor, int n)  
{  
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    int i, j, min_idx;  

    // One by one move boundary of unsorted subarray  
    for (i = 0; i < n-1; i++)  
    {  
        // Find the minimum element in unsorted array  
        min_idx = i;  
        for (j = i+1; j < n; j++)  
        if (tensor[j] < tensor[min_idx])  
            min_idx = j;  
  
        // Swap the found minimum element with the first element  
        float tmp = tensor[min_idx];
        tensor[min_idx] = tensor[i];
        tensor[i] = tmp; 
    }  

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>( t2 - t1 ).count();
    cout << "Execution Time (" << __func__ << ") (microseconds): "<< duration << endl;
}  

void SelectionSortDescending(float* tensor, int n)  
{  
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    int i, j, max_idx;  

    // One by one move boundary of unsorted subarray  
    for (i = 0; i < n-1; i++)  
    {  
        // Find the maximum element in unsorted array  
        max_idx = i;  
        for (j = i+1; j < n; j++)  
        if (tensor[j] > tensor[max_idx])  
            max_idx = j;  
  
        // Swap the found maximum element with the first element  
        float tmp = tensor[max_idx];
        tensor[max_idx] = tensor[i];
        tensor[i] = tmp; 
    }  

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>( t2 - t1 ).count();
    cout << "Execution Time (" << __func__ << ") (microseconds): "<< duration << endl;
}  

//only first k elements are valid
void SelectionSortTopK(float* tensor, int n, int k)  
{  
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    int i, j, max_idx;  

    // One by one move boundary of unsorted subarray  
    for (i = 0; i < k; i++)  
    {  
        // Find the maximum element in unsorted array  
        max_idx = i;  
        for (j = i+1; j < n; j++)  
        if (tensor[j] > tensor[max_idx])  
            max_idx = j;  
  
        // Swap the found maximum element with the first element  
        float tmp = tensor[max_idx];
        tensor[max_idx] = tensor[i];
        tensor[i] = tmp; 
    }  

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>( t2 - t1 ).count();
    cout << "Execution Time (" << __func__ << ") (microseconds): "<< duration << endl;
}  

void PrintTensor(const float *tensor, int len, string strname){
	cout<<endl<<endl<<"DUMP: "<< strname<<endl;
	for(int i=0;i<len;i++){
		cout<< strname << "["<< i<<"] = "<<tensor[i]<<endl;
	}
	cout<<endl;
}

bool CompareTensors(const float *tensorGold, const float *tensorTest, int len){
	bool rslt=true;
	for(int i=0;i<len;i++){
		if(tensorGold[i] != tensorTest[i]){
			rslt = false;
		}
	}
	return rslt;
}

int main(){
	cout<<""<<endl;
	float inputTn[CONFIG_LEN];
	float sortedTnAscending[CONFIG_LEN];
	float sortedTnDescending[CONFIG_LEN];
	float topkTn[CONFIG_LEN];

	InitTensor(inputTn,CONFIG_LEN,0);

	CloneTensor(inputTn,sortedTnAscending,CONFIG_LEN);
	SelectionSortAscending(sortedTnAscending,CONFIG_LEN);

	CloneTensor(inputTn,sortedTnDescending,CONFIG_LEN);
	SelectionSortDescending(sortedTnDescending,CONFIG_LEN);

	CloneTensor(inputTn,topkTn,CONFIG_LEN);
	SelectionSortTopK(topkTn,CONFIG_LEN,CONFIG_K);

	PrintTensor(inputTn,CONFIG_LEN,"inputTn");
	PrintTensor(sortedTnAscending,CONFIG_LEN,"sortedTnAscending");
	PrintTensor(sortedTnDescending,CONFIG_LEN,"sortedTnDescending");
	PrintTensor(topkTn,CONFIG_K,"topkTn");

	cout<<"TopK results are " << (CompareTensors(sortedTnDescending,topkTn,CONFIG_K) ? "" : "NOT") << "valid." << endl;
}
