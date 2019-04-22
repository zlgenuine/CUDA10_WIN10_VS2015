#include "cuda_runtime.h"
#include "cublas_v2.h"

#include <time.h>
#include <iostream>

using namespace std;

//define a Matrix's dimensionality

int const M = 5;
int const N = 10;

int main()
{
	//declare a state variable 

	cublasStatus_t status;

	//assign space for defined matrix
	float *h_A = (float*)malloc(N*M * sizeof(float));
	float *h_B = (float*)malloc(N*M * sizeof(float));

	//assign space for obtained result
	float *h_C = (float*)malloc(N*M * sizeof(float));

	//randomly generate a number from the range [0,10] for each cell of matrix
	for (int i = 0; i < N*M; i++) {
		h_A[i] = (float)(rand() % 10 + 1);
		h_B[i] = (float)(rand() % 10 + 1);
	}

	// print the input matrix 
	cout << "Matrix A:" << endl;
	for (int i = 0; i < N*M; i++) {
		cout << h_A[i] << " ";
		if ((i + 1) % N == 0) cout << endl;
	}
	cout << endl;
	cout << "Matrix B:" << endl;
	for (int i = 0; i < N*M; i++) {
		cout << h_B[i] << " ";
		if ((i + 1) % N == 0) cout << endl;
	}
	cout << endl;

	/*
	**matrix product using GPU
	*/

	// create CUBLAS library object and initialise it
	cublasHandle_t handle;
	status = cublasCreate(&handle);

	if (status != CUBLAS_STATUS_SUCCESS) {
		
		if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
			cout << "initialisation error of cublas object" << endl;
		}
		getchar();
		return EXIT_FAILURE;
	}

	float *d_A, *d_B, *d_C;
	//allocate matrix space in GPU
	cudaMalloc(
		(void**)&d_A, // the head of pointer
		N*M * sizeof(float) //assign the number of Bytes required
	);

	cudaMalloc(
		(void**)&d_B,
		N*M * sizeof(float)
	);

	//allocate result matrix in GPU
	cudaMalloc(
		(void**)&d_C,
		M*M * sizeof(float)
	);


	cublasSetVector(
		N*M, //the number of elements 
		sizeof(float), //the size of each element
		h_A, //the start point of GPU address of pointer h_A
		1,	//space break by 1 between two adjacent elements
		d_A, //the start point of GPU address of pointer d_A
		1 //space break by 1 between two adjacent elements
	);

	cublasSetVector(
		N*M,
		sizeof(float),
		h_B,
		1,
		d_B,
		1
	);

	//synchronization function
	cudaThreadSynchronize();

	//// 传递进矩阵相乘函数中的参数，具体含义请参考函数手册。  
	float a = 1; float b = 0;
	// 矩阵相乘。该函数必然将数组解析成列优先数组  
	cublasSgemm(
		handle,    // blas 库对象   
		CUBLAS_OP_T,    // 矩阵 A 属性参数  
		CUBLAS_OP_T,    // 矩阵 B 属性参数  
		M,    // A, C 的行数   
		M,    // B, C 的列数  
		N,    // A 的列数和 B 的行数  
		&a,    // 运算式的 α 值  
		d_A,    // A 在显存中的地址  
		N,    // lda  
		d_B,    // B 在显存中的地址  
		M,    // ldb  
		&b,    // 运算式的 β 值  
		d_C,    // C 在显存中的地址(结果矩阵)  
		M    // ldc  
	);
	// 同步函数  
	cudaThreadSynchronize();

	// 从 显存 中取出运算结果至 内存中去  
	cublasGetVector(
		M*M,    //  要取出元素的个数  
		sizeof(float),    // 每个元素大小  
		d_C,    // GPU 端起始地址  
		1,    // 连续元素之间的存储间隔  
		h_C,    // 主机端起始地址  
		1    // 连续元素之间的存储间隔  
	);

	// 打印运算结果  
	cout << "(A*B)^T" << endl;
	for (int i = 0; i < M*M; i++) {
		cout << h_C[i] << " ";
		if ((i + 1) % M == 0) cout << endl;
	}

	// 清理掉使用过的内存  
	free(h_A);
	free(h_B);
	free(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	// 释放 CUBLAS 库对象  
	cublasDestroy(handle);

	getchar();

	return 0;
}