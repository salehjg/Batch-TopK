#include <iostream>
#include <string>
#include <stdlib.h>
#include <chrono>
#include <assert.h>
using namespace std;
using namespace std::chrono;

#define CONFIG_LEN (1024)
#define CONFIG_K   (20)

#define CONFIG_B (5)
#define CONFIG_N (1024)


float RandomFloat(float min, float max){
    float scale = rand() / (float) RAND_MAX; /* [0, 1.0] */
    return min + scale * ( max - min );      /* [min, max] */
}

/*
void InitTensor(float *tensor, int len, int mode){
	if(mode==0){
		for(int i=0;i<len;i++){
			tensor[i] = RandomFloat(-2.0f,2.0f);
		}
	}
}
*/

void InitTensor(float *tensor, int dim0, int dim1, int dim2, int mode){
    unsigned long indxD;
    
    for(int d0=0; d0<dim0; d0++){
        for(int d1=0; d1<dim1; d1++){
            for(int d2=0; d2<dim2; d2++){
                indxD = d0 * dim1*dim2 +
                        d1 * dim2 +
                        d2;
                if(mode==0){
                    tensor[indxD] = RandomFloat(-2.0f,2.0f);
                }else if(mode==1){
                    tensor[indxD] = d2;
                }else if(mode==2){
                    tensor[indxD] = 10*(d1*d0) + d2;
                }
            }
        }
    }
    
}

/*
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
        for (j = i+1; j < n; j++){
            if (tensor[j] < tensor[min_idx])  
                min_idx = j;  
        }
  
        // Swap the found minimum element with the first element  
        if(min_idx != i){
            float tmp = tensor[min_idx];
            tensor[min_idx] = tensor[i];
            tensor[i] = tmp; 
        }
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
        for (j = i+1; j < n; j++){
            if (tensor[j] > tensor[max_idx])  
                max_idx = j;  
        }
  
        // Swap the found maximum element with the first element  
        if(min_idx != i){
            float tmp = tensor[max_idx];
            tensor[max_idx] = tensor[i];
            tensor[i] = tmp; 
        }
    }  

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>( t2 - t1 ).count();
    cout << "Execution Time (" << __func__ << ") (microseconds): "<< duration << endl;
} 
*/ 

/*
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
        for (j = i+1; j < n; j++){
            if (tensor[j] > tensor[max_idx])  
                max_idx = j;  
        }

        // Swap the found maximum element with the first element  
        if(min_idx != i){
            float tmp = tensor[max_idx];
            tensor[max_idx] = tensor[i];
            tensor[i] = tmp; 
        }
    }  

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>( t2 - t1 ).count();
    cout << "Execution Time (" << __func__ << ") (microseconds): "<< duration << endl;
}  
*/

/*
inputTn:    INPUT - Distance tensor of rank three (dim0xdim1xdim2), row-major
indicesTn:  OUTPUT- Indices of top 'k' elements for each dim2 slice of inputTn (dim0xdim1xK)
outputTn:   OUTPUT- Fully sorted version of inputTn (dim0xdim1xdim2)
*/
void BatchSelectionSortTopK(
    const float* inputTn,
    int* indicesTn,
    float* outputTn,
    int dim0,
    int dim1,
    int dim2,
    int kValue){

    int i, j, max_idx;  
    
    assert(kValue<dim2);

    // 1. Copy inputTn into outputTn, so sorting algorithm could be
    //    run on outputTn without editing inputTn.
    for(unsigned long i = 0; i<dim0*dim1*dim2; i++){
        outputTn[i] = inputTn[i];
    }

    // 2. Initializing indicesTn for each of k-element slices of it.
    for(int batch=0; batch<dim0*dim1; batch++){
        for(int i=0; i<kValue; i++){
            indicesTn[batch*kValue + i] = i;
        }
    }

    // 3. Running descending selection sort only for first k elements of
    //    each of dim2 slices of outputTn(which is a clone of inputTn).
    for(int batch=0; batch<dim0*dim1; batch++){

        // Run selection sort on current slice of dim2.
        for (i = 0; i < kValue; i++)  
        {  
            // Find the maximum element in unsorted array  
            max_idx = i;  
            for (j = i+1; j < dim2; j++){
                if (outputTn[batch*kValue + j] > outputTn[batch*kValue + max_idx])  
                    max_idx = j;  
            }
            
            // Swap the found maximum element with the first element  
            if(max_idx != i){
                float tmp = outputTn[batch*kValue + max_idx];
                outputTn[batch*kValue + max_idx] = outputTn[batch*kValue + i];
                outputTn[batch*kValue + i] = tmp; 
                //------------------------------------------------------------
                int tmpi = indicesTn[batch*kValue + max_idx];
                indicesTn[batch*kValue + max_idx] = indicesTn[batch*kValue + i];
                indicesTn[batch*kValue + i] = tmpi;
            }
        } 

    }
}

/*
void PrintTensor(const float *tensor, int len, string strname){
	cout<<endl<<endl<<"DUMP: "<< strname<<endl;
	for(int i=0;i<len;i++){
		cout<< strname << "["<< i<<"] = "<<tensor[i]<<endl;
	}
	cout<<endl;
}
*/

void PrintTensor(
    const float *tensor,
    int dim0,
    int dim1,
    int dim2,
    string strname,
    int limitDim0=1,
    int limitDim1=10){

    assert(limitDim0<dim0);
    assert(limitDim1<dim1);

    unsigned long indxD;

    cout<<endl<<endl<<"DUMP: "<< strname<<endl;
    for(int d0=0; d0<limitDim0; d0++){
        for(int d1=0; d1<limitDim1; d1++){
            for(int d2=0; d2<dim2; d2++){
                indxD = d0*dim1*dim2 + d1*dim2 + d2;
                cout<< strname << "[d0,d1,d2:"<< d0 << ", " << d1 << ", " << d2 << "] = " << tensor[indxD] << endl;
            }
        }
    }

    cout<<endl;
}

/*
bool CompareTensors(const float *tensorGold, const float *tensorTest, int len){
	bool rslt=true;
	for(int i=0;i<len;i++){
		if(tensorGold[i] != tensorTest[i]){
			rslt = false;
		}
	}
	return rslt;
}
*/

/*
int mainOld(){
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
*/

int main(){
    float distanceTn[CONFIG_B * CONFIG_N * CONFIG_N];
    float sortedTn[CONFIG_B * CONFIG_N * CONFIG_N]; 
    int indicesTn[CONFIG_B * CONFIG_N * CONFIG_K];

    // 1. Init batch of distance matrices of NxN. (BxNxN)
    InitTensor(distanceTn, CONFIG_B, CONFIG_N, CONFIG_N, 2);

    // 2. Run batch-topk op on distance tensor.
    BatchSelectionSortTopK(distanceTn, indicesTn, sortedTn, CONFIG_B, CONFIG_N, CONFIG_N, CONFIG_K);

    // 3. Dumping some of the sortedTn dim2 slices for the user.
    PrintTensor(sortedTn, CONFIG_B, CONFIG_N, CONFIG_N, "sortedTn");
}