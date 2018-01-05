#ifndef RBF_H_
#define RBF_H_

typedef struct ndata_set* ndata_object;

ndata_object ndata_init(int inputs, int outputs, int patterns);

struct ndata_set {
	int I;
	int O;
	int P;
	int tsize;
	int gsize;
	int vsize;
	double *data;
	double *target;
	double params[1];
};

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


void rbfnet_set_training_ratios(rbfnet_object obj, double tratio, double vratio);

void rbfnet_train(rbfnet_object obj, int size, double *data, double *target);

void erbfnet_train(erbfnet_object obj, int size, double *data, double sigma, double *target);

void rbfnet_sim(rbfnet_object obj, int size, double *data, double *output);

void erbfnet_sim(erbfnet_object obj, int size, double *data, double *output);

void rbfnet_free(rbfnet_object obj);

void erbfnet_free(erbfnet_object obj);

void interleave(double *inp, int size, int M, double *oup);

void data_enter(ndata_object obj, double *data, double *target);

void data_interleave_enter(ndata_object obj, double *data, double *target);

void csvreader(ndata_object obj, const char *filepath, const char *delimiter, int isHeader);

void file_enter(ndata_object obj, const char *filepath, const char *delimiter, int isHeader);

void file_rev_enter(ndata_object obj, const char *filepath, const char *delimiter, int isHeader);

void file_sep_line_enter(ndata_object obj, const char *filepath, const char *delimiter, int isHeader);

void ndata_check(ndata_object obj);

void ndata_free(ndata_object obj);

#endif /* RBF_H_ */