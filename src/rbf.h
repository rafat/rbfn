#ifndef RBF_H_
#define RBF_H_

#include "kmeans.h"

typedef struct rbfnet_set* rbfnet_object;

rbfnet_object rbfnet_init(int I, int O, int H);

struct rbfnet_set {
	int I;// Number of Inputs
	int H;// Number of Hidden Layers
	int O;// Number of Outputs
	int normmethod;// 0 - NULL, 1 - Minmax {-1,1}, 2 - Std (Mean = 0, Variance = 1}. 
	int lw;
	double tratio;
	double vratio;
	double beta;
	double mse;
	char beta_type[20];
	double *weight;
	double *centroid;
	double params[1];
};

typedef struct erbfnet_set* erbfnet_object;

erbfnet_object erbfnet_init(int I, int O, int datasize);

struct erbfnet_set {
	int I;// Number of Inputs
	int H;// Number of Hidden Layers
	int O;// Number of Outputs
	int normmethod;// 0 - NULL, 1 - Minmax {-1,1}, 2 - Std (Mean = 0, Variance = 1}.
	int lw;
	double beta;
	char beta_type[20];
	double *weight;
	double *centroid;
	double params[1];
};

typedef struct orbfnet_set* orbfnet_object;

orbfnet_object orbfnet_init(int I, int O, int hmax);

struct orbfnet_set {
	int I;// Number of Inputs
	int H;// Number of Hidden Layers
	int O;// Number of Outputs
	int normmethod;// 0 - NULL, 1 - Minmax {-1,1}, 2 - Std (Mean = 0, Variance = 1}. 
	int lw;
	int hmax;
	int scheck;
	double tratio;
	double vratio;
	double beta;
	double mse;
	double stol;
	char beta_type[20];
	double *weight;
	double *centroid;
	double params[1];
};

int lls_svd_p(double *A, double *b, int M, int N, int p, double *x);

int lls_svd_p_transpose(double *A, double *b, int M, int N, int p, double *x);

int lls_qr_p(double *Ai, double *bi, int M, int N, int p, double *xo);

void rbfnet_set_training_ratios(rbfnet_object obj, double tratio, double vratio);

void rbfnet_train(rbfnet_object obj, int size, double *data, double *target);

void erbfnet_train(erbfnet_object obj, int size, double *data, double sigma, double *target);

void orbfnet_train(orbfnet_object obj, int size, double *data, double sigma, double *target);

void rbfnet_sim(rbfnet_object obj, int size, double *data, double *output);

void erbfnet_sim(erbfnet_object obj, int size, double *data, double *output);

void orbfnet_sim(orbfnet_object obj, int size, double *data, double *output);

void rbfnet_free(rbfnet_object obj);

void erbfnet_free(erbfnet_object obj);

void orbfnet_free(orbfnet_object obj);

#endif /* RBF_H_ */