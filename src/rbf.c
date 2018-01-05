#include "rbf.h"

#ifndef _OPENMP
#define omp_get_wtime() 0
#endif

rbfnet_object rbfnet_init(int I, int O, int H) {
	rbfnet_object obj = NULL;
	int lw,i,tvector,cw;

	lw = (H + 1) * O;
	cw = I * H;
	tvector = lw + cw;
	obj = (rbfnet_object)malloc(sizeof(struct rbfnet_set) + sizeof(double)* tvector);

	obj->tratio = 1.0;
	obj->vratio = 0.0;
	obj->normmethod = 0;
	obj->lw = lw;
	obj->I = I;
	obj->H = H;
	obj->O = O;
	for (i = 0; i < tvector; ++i) {
		obj->params[i] = 0.0;
	}

	obj->weight = &obj->params[0];
	obj->centroid = &obj->params[lw];

	return obj;
}

erbfnet_object erbfnet_init(int I, int O, int datasize) {
	erbfnet_object obj = NULL;
	int lw, i, tvector, cw;

	lw = (datasize + 1) * O;
	cw = I * datasize;
	tvector = lw + cw;
	obj = (erbfnet_object)malloc(sizeof(struct erbfnet_set) + sizeof(double)* tvector);

	obj->normmethod = 0;
	obj->lw = lw;
	obj->I = I;
	obj->H = datasize;
	obj->O = O;
	for (i = 0; i < tvector; ++i) {
		obj->params[i] = 0.0;
	}

	obj->weight = &obj->params[0];
	obj->centroid = &obj->params[lw];

	return obj;
}

orbfnet_object orbfnet_init(int I, int O, int hmax) {
	orbfnet_object obj = NULL;
	
	int lw, i, tvector, cw;

	lw = (hmax + 1) * O;
	cw = I * hmax;
	tvector = lw + cw;
	obj = (orbfnet_object)malloc(sizeof(struct orbfnet_set) + sizeof(double)* tvector);

	obj->tratio = 1.0;
	obj->vratio = 0.0;
	obj->normmethod = 0;
	obj->lw = lw;
	obj->I = I;
	obj->hmax = hmax;
	obj->O = O;
	obj->scheck = 1;
	obj->stol = 1.0e-10; // default
	obj->mse = 0.0;
	for (i = 0; i < tvector; ++i) {
		obj->params[i] = 0.0;
	}

	obj->weight = &obj->params[0];
	obj->centroid = &obj->params[lw];

	return obj;
}

static double distmax(double *centroid,int K,int d) {
	double dmax,dist;
	int i, j;

	dmax = 0.0;

	for (i = 0; i < K - 1; ++i) {
		for (j = i + 1; j < K; ++j) {
			dist = l2(centroid + i*d, centroid + j*d, d);
			if (dist > dmax) {
				dmax = dist;
			}
		}
	}

	return dmax;
}

static double distavg(double *centroid, int K, int d) {
	double davg, dist;
	int i, j,acc;

	davg = 0.0;
	acc = 0;
	for (i = 0; i < K - 1; ++i) {
		for (j = i + 1; j < K; ++j) {
			dist = l2(centroid + i*d, centroid + j*d, d);
			davg += dist;
			acc++;
		}
	}

	davg /= (double)acc;

	return davg;
}

void rbfnet_set_training_ratios(rbfnet_object obj, double tratio, double vratio) {
	if (tratio + vratio != 1.0) {
		printf("Ratios must sum to 1.0\n");
		exit(1);
	}

	obj->tratio = tratio;
	obj->vratio = vratio;
}

int lls_svd_p(double *A, double *b, int M, int N,int p, double *x) {
	int rnk, ret, i,j;
	double *U, *V, *q, *d;
	double eps, tol, szmax, qmax;

	if (M < N) {
		printf(" M < N . Use method lls_svd_p_transpose.\n");
		exit(-1);
	}

	U = (double*)malloc(sizeof(double)* M*N);
	V = (double*)malloc(sizeof(double)* N*N);
	q = (double*)malloc(sizeof(double)* N);
	/*
	The code returns -1 if SVD computation fails else it returns the rank of the matrix A (and the real size of vector x)
	*/
	ret = svd(A, M, N, U, V, q);

	if (ret != 0) {
		printf("Failed to Compute SVD");
		return -1;
	}

	szmax = (double)M;

	eps = macheps();
	rnk = 0;

	qmax = q[0];

	tol = qmax*szmax *eps;

	for (i = 0; i < N; ++i) {
		if (q[i] > tol) {
			rnk++;
		}
	}

	mtranspose(U, M, N, A);

	d = (double*)malloc(sizeof(double)* N * p);

	mmult(A, b, d, N, M, p);

	for (j = 0; j < p; ++j) {
		for (i = 0; i < rnk; ++i) {
			d[i*p+j] /= q[i];
		}

		for (i = rnk; i < N; ++i) {
			d[i*p+j] = 0.0;
		}
	}
	mmult(V, d, x, N, N, p);

	free(U);
	free(V);
	free(q);
	free(d);

	return(rnk);
}

int lls_qr_p(double *Ai, double *bi, int M, int N,int p, double *xo) {
	int j, i, k, u, t, retcode, c1, l,h,t2;
	double *x, *v, *AT, *w, *bvec, *b, *A, *R;
	double beta, sum;

	retcode = 0;

	if (M < N) {
		printf("M should be greater than or equal to N");
		exit(-1);
	}

	x = (double*)malloc(sizeof(double)* M);
	b = (double*)malloc(sizeof(double)* M*p);
	bvec = (double*)malloc(sizeof(double)* N);
	v = (double*)malloc(sizeof(double)* M);
	AT = (double*)malloc(sizeof(double)* M * N);
	A = (double*)malloc(sizeof(double)* M * N);
	w = (double*)malloc(sizeof(double)* N);
	R = (double*)malloc(sizeof(double)* N * N);


	mtranspose(bi, M, p, b);
	for (j = 0; j < M*N; ++j) {
		A[j] = Ai[j];
	}

	for (j = 0; j < N; ++j) {
		for (i = j; i < M; ++i) {
			x[i - j] = A[i*N + j];

		}

		beta = house(x, M - j, v);
		bvec[j] = beta;

		for (i = j; i < M; i++) {
			t = i * N;
			u = 0;
			for (k = j; k < N; k++) {
				AT[u + i - j] = A[k + t];
				u += (M - j);

			}

		}


		mmult(AT, v, w, N - j, M - j, 1);
		scale(w, N - j, 1, beta);
		mmult(v, w, AT, M - j, 1, N - j);
		for (i = j; i < M; i++) {
			t = i *N;
			for (k = j; k < N; k++) {
				A[t + k] -= AT[(i - j)*(N - j) + k - j];
			}
		}
		if (j < M) {

			for (i = j + 1; i < M; ++i) {
				A[i*N + j] = v[i - j];
			}
		}

	}

	for (i = 0; i < N; ++i) {
		t = i *N;
		for (j = 0; j < N; ++j) {
			if (i > j) {
				R[t + j] = 0.;
			}
			else {
				R[t + j] = A[t + j];
			}
		}
	}

	for (h = 0; h < p; ++h) {
		t = h*M;
		for (j = 0; j < N; ++j) {
			v[j] = 1;
			for (i = j + 1; i < M; ++i) {
				v[i] = A[i * N + j];//edit
			}
			mmult(b + t + j, v + j, w, 1, M - j, 1);
			*w = *w * bvec[j];
			for (i = j; i < M; ++i) {
				v[i] = *w * v[i];
			}
			for (i = j; i < M; ++i) {
				b[t+i] = b[t+i] - v[i];
			}
		}
	}

	

	//mdisplay(b,1,M);

	//back substitution

	
	for (h = 0; h < p; ++h) {
		t = h *N;
		t2 = h *M;
		xo[t+N - 1] = b[t2+N - 1] / R[N * N - 1];

		for (i = N - 2; i >= 0; i--) {
			sum = 0.;
			c1 = i*(N + 1);
			l = 0;
			for (j = i + 1; j < N; j++) {
				l++;
				sum += R[c1 + l] * xo[t+j];
			}
			xo[t+i] = (b[t2+i] - sum) / R[c1];
		}
	}

	itranspose(xo, p, N);

	free(x);
	free(v);
	free(AT);
	free(w);
	free(bvec);
	free(R);
	free(b);
	free(A);

	return retcode;
}

int lls_svd_p_transpose(double *A, double *b, int M, int N, int p, double *x) {
	int rnk, ret, i, j;
	double *U, *V, *q, *d;
	double eps, tol, szmax, qmax;

	if (M >= N) {
		printf(" M >= N . Use method lls_svd_p.\n");
		exit(-1);
	}

	U = (double*)malloc(sizeof(double)* M*M);
	V = (double*)malloc(sizeof(double)* N*M);
	q = (double*)malloc(sizeof(double)* M);
	/*
	The code returns -1 if SVD computation fails else it returns the rank of the matrix A (and the real size of vector x)
	*/

	//mdisplay(A, M, N);
	ret = svd_transpose(A, M, N, U, V, q);

	if (ret != 0) {
		printf("Failed to Compute SVD");
		return -1;
	}

	szmax = (double)N;

	eps = macheps();
	rnk = 0;

	qmax = q[0];

	tol = qmax*szmax *eps;

	for (i = 0; i < M; ++i) {
		if (q[i] > tol) {
			rnk++;
		}
	}

	mtranspose(U, M, M, A);

	d = (double*)malloc(sizeof(double)* M * p);

	mmult(A, b, d, M, M, p);

	for (j = 0; j < p; ++j) {
		for (i = 0; i < rnk; ++i) {
			d[i*p + j] /= q[i];
		}

		for (i = rnk; i < M; ++i) {
			d[i*p + j] = 0.0;
		}
	}
	mmult(V, d, x, N, M, p);

	free(U);
	free(V);
	free(q);
	free(d);

	return(rnk);
}

static void actfcn_gauss(double *data,int N,int d,double *centroid,int K,double beta,double *A) {
	int i, j, K1,j1,it1,ik1;
	double dist;
	K1 = K + 1;

	for (i = 0; i < N; ++i) {
		ik1 = i * K1;
		A[ik1] = 1.0;
		it1 = i * d;
		for (j = 1; j < K1; ++j) {
			j1 = j - 1;
			dist = l2s(data + it1, centroid + j1*d, d);
			A[ik1 + j] = exp(-beta *dist);
		}
	}
}

static void actfcn_gauss2(double *data, int N, int d, double *centroid, int K, double beta, double *A) {
	int i, j, it1, ik1;
	double dist;

	for (i = 0; i < N; ++i) {
		ik1 = i * K;
		it1 = i * d;
		for (j = 0; j < K; ++j) {
			dist = l2s(data + it1, centroid + j*d, d);
			A[ik1 + j] = exp(-beta *dist);
		}
	}
}

static void actfcn_gauss3(double *data, int N, int d, double *centroid, int K, double beta, double *A) {
	int i, j, K1, j1, it1, ik1;
	double dist, t;
	K1 = K + 1;

	for (i = 0; i < N; ++i) {
		ik1 = i * K1;
		A[ik1] = 1.0;
		it1 = i * d;
		//printf("\n");
		for (j = 1; j < K1; ++j) {
			j1 = j - 1;
			dist = l2(data + it1, centroid + j1*d, d);
			//printf("%g ", dist);
			t = beta *dist;
			A[ik1 + j] = exp(-t*t);
		}
	}
}

static void actfcn_gauss4(double *data, int N, int d, double *centroid, int K, double beta, double *A) {
	int i, j, it1, ik1;
	double dist, t;

	for (i = 0; i < N; ++i) {
		ik1 = i * K;
		it1 = i * d;
		//printf("\n");
		for (j = 0; j < K; ++j) {
			dist = l2(data + it1, centroid + j*d, d);
			//printf("%g ", dist);
			t = beta *dist;
			A[ik1 + j] = exp(-t*t);
		}
	}
}

static void actfcn_gauss6(double *data, int N, int d, double *centroid, double beta, double *A) {
	int i, it1;
	double dist, t;

	for (i = 0; i < N; ++i) {
		it1 = i * d;
		dist = l2s(data + it1, centroid, d);
		t = beta *beta;
		A[i] = exp(-t*dist);
	}
}

static void rbf_sim(int size, double *data,int leninp,int lenoup,int clusters,double beta,int lw, double *weight, double *centroid, double *output) {
	int K1;
	double *A, *x;

	K1 = clusters + 1;

	A = (double*)malloc(sizeof(double)* size * (clusters + 1));
	x = (double*)malloc(sizeof(double)* lw);

	actfcn_gauss(data, size, leninp, centroid, clusters, beta, A);

	mtranspose(weight, lenoup, K1, x);

	mmult(A, x, output, size, K1, lenoup);

	free(A);
	free(x);
}

static void orbf_sim(int size, double *data, int leninp, int lenoup, int clusters, double beta, int lw, double *weight, double *centroid, double *output) {
	int K1;
	double *A, *x;

	K1 = clusters + 1;

	A = (double*)malloc(sizeof(double)* size * (clusters + 1));
	x = (double*)malloc(sizeof(double)* lw);

	actfcn_gauss3(data, size, leninp, centroid, clusters, beta, A);

	mtranspose(weight, lenoup, K1, x);

	mmult(A, x, output, size, K1, lenoup);

	printf("\n\n OK \n\n");

	free(A);
	free(x);
}


void rbfnet_sim(rbfnet_object obj, int size, double *data, double *output) {
	int leninp, lenoup, clusters,lw;
	double beta;

	leninp = obj->I;
	lenoup = obj->O;
	clusters = obj->H;
	beta = obj->beta;
	lw = obj->lw;

	rbf_sim(size, data, leninp, lenoup, clusters, beta, lw, obj->weight, obj->centroid, output);
	
}

void erbfnet_sim(erbfnet_object obj, int size, double *data, double *output) {
	int leninp, lenoup, clusters, lw;
	double beta;

	leninp = obj->I;
	lenoup = obj->O;
	clusters = obj->H;
	beta = obj->beta;
	lw = obj->lw;

	rbf_sim(size, data, leninp, lenoup, clusters, beta, lw, obj->weight, obj->centroid, output);

}

void orbfnet_sim(orbfnet_object obj, int size, double *data, double *output) {
	int leninp, lenoup, clusters, lw;
	double beta;

	leninp = obj->I;
	lenoup = obj->O;
	clusters = obj->H;
	beta = obj->beta;
	lw = obj->lw;

	orbf_sim(size, data, leninp, lenoup, clusters, beta, lw, obj->weight, obj->centroid, output);

	/*
	K1 = clusters + 1;

	A = (double*)malloc(sizeof(double)* size * (clusters + 1));
	x = (double*)malloc(sizeof(double)* lw);
	//rbf_sim(size, data, leninp, lenoup, clusters, beta, lw, obj->weight, obj->centroid, output);

	actfcn_gauss3(data, size, leninp, obj->centroid, clusters, beta, A);

	mtranspose(obj->weight, lenoup, K1, x);

	mmult(A, x, output, size, K1, lenoup);
	
	free(A);
	free(x);
	*/
}

static double rbfnet_validate(rbfnet_object obj, int tsize, double *data, double *target) {
	double gmse, temp;
	int i, itrd, itrt, leninp, lenoup, j;
	double *output;

	leninp = obj->I;
	lenoup = obj->O;
	gmse = 0.0;

	output = (double*)malloc(sizeof(double)* tsize * lenoup);

	rbfnet_sim(obj, tsize, data, output);

	for (i = 0; i < tsize; ++i) {
		itrd = i * leninp;
		itrt = i * lenoup;
		for (j = 0; j < lenoup; ++j) {
			temp = target[itrt + j] - output[j];
			gmse += (temp*temp);
		}
	}

	gmse = gmse / (lenoup * tsize);

	free(output);

	return gmse;
}

void rbfnet_train(rbfnet_object obj, int size, double *data, double *target) {
	int d,clusters,p,lw,rnk,K1;
	int tsize, vsize;
	int *clusterindex;
	double sigma,beta,gmse;
	double *A,*x;
	int itrt, itrd,val;

	tsize = (int)(obj->tratio * size); // training size
	vsize = size - tsize; // validation size

	clusters = obj->H;
	d = obj->I;
	p = obj->O;
	lw = obj->lw;
	clusterindex = (int*)malloc(sizeof(int)* tsize);
	A = (double*)malloc(sizeof(double)* tsize * (clusters+1));
	x = (double*)malloc(sizeof(double)* lw);

	itrd = tsize * d;
	itrt = tsize * p;
	val = 0;

	if (vsize > 0) {
		val = 1;
	}

	// Find Hidden Unit centroids

	kmeans(data, tsize,d, clusters, obj->centroid,clusterindex);

	// Find beta

	sigma = distmax(obj->centroid, clusters, d);

	sigma /= sqrt(2 * clusters);

	beta = 1.0 / (2.0 * sigma * sigma);

	obj->beta = beta;

	// Gaussian Activation

	actfcn_gauss(data,tsize,d,obj->centroid,clusters,beta,A);

	K1 = clusters + 1;

	if (tsize >= K1) {
		rnk = lls_svd_p(A, target, tsize, K1, p, x);
	}
	else {
		rnk = lls_svd_p_transpose(A, target, tsize, K1, p, x);
	}

	if (rnk == -1) {
		printf("SVD Computation Fails \n");
		exit(-1);
	}
	else {
		mtranspose(x, K1, p, obj->weight);
	}

	if (val == 1) {
		gmse = rbfnet_validate(obj, vsize, data + itrd, target + itrt);
		printf("Error %g \n", gmse);
	}

	free(clusterindex);
	free(A);
	free(x);
}

void erbfnet_train(erbfnet_object obj, int size, double *data, double sigma, double *target) {
	int d, clusters, p, lw, rnk, K1,i;
	double beta;
	double *A, *x;
	clusters = obj->H;
	d = obj->I;
	p = obj->O;
	lw = obj->lw;

	if (size != obj->H) {
		printf("Data Size should be equal to Hidden Neurons %d \n", obj->H);
		exit(1);
	}

	A = (double*)malloc(sizeof(double)* size * (clusters + 1));
	x = (double*)malloc(sizeof(double)* lw);

	//Set Centroids

	for (i = 0; i < d * size; ++i) {
		obj->centroid[i] = data[i];
	}

	beta = 1.0 / (2.0 * sigma * sigma);

	obj->beta = beta;

	// Gaussian Activation

	actfcn_gauss(data, size, d, obj->centroid, clusters, beta, A);

	K1 = clusters + 1;

	if (size >= K1) {
		rnk = lls_svd_p(A, target, size, K1, p, x);
	}
	else {
		rnk = lls_svd_p_transpose(A, target, size, K1, p, x);
	}

	if (rnk == -1) {
		printf("SVD Computation Fails \n");
		exit(-1);
	}
	else {
		mtranspose(x, K1, p, obj->weight);
	}

	free(A);
	free(x);
}

static int max_error(double *err, int M, int O,double *errsum) {
	int index, i, j, itr;

	double sum, max;

	max = 0.0;

	index = 0;
	itr = 0;

	for (i = 0; i < M; ++i) {
		sum = 0.0;

		for (j = 0; j < O; ++j) {
			if (err[itr + j] == err[itr + j]) {
				sum += (err[itr + j] * err[itr + j]);
			}
		}

		errsum[i] = sum;

		if (sum >= max) {
			max = sum;
			index = i;
		}

		itr += O;
	}

	return index;
}

static int ols_cgs(double *data, int N, int I,int M, int O, double beta, double stol,int scheck, double *weight,double *centroid, double *target) {
	int i, j, k, l, itr, itr2, itr3,itr4, index, m2, count,clusters,K1,rnk,ind,icount,lwt;
	int *check,*index2;
	double *err, *W, *w1, *g, *dt, *dtd, *dd, *A, *num, *den, *P,*errsum,*W2,*output;
	double w1t, alpha,ssqr,mse;
	double tsvd, tloop, t0, t1;

	dt = (double*)malloc(sizeof(double)* O * N);
	dtd = (double*)malloc(sizeof(double)* O * O);

	dd = (double*)malloc(sizeof(double)* O);

	mtranspose(target, N, O, dt);

	mmult(dt, target, dtd, O, N, O);

	free(dt);

	for (i = 0; i < O; ++i) {
		dd[i] = 0.0;
	}
	itr = 0;
	for (i = 0; i < O; ++i) {
		for (j = 0; j < O; ++j) {
			dd[j] += dtd[itr + j];
		}
		itr += O;
	}
	
	for (i = 0; i < O; ++i) {
		if (dd[i] == 0) {
			dd[i] = DBL_MAX;
		}
	}

	free(dtd);
	itr = 0;
	m2 = (M * (M - 1) / 2);
	err = (double*)malloc(sizeof(double)* N * O);
	output = (double*)malloc(sizeof(double)* N * O);
	w1 = (double*)malloc(sizeof(double)* N);
	g = (double*)malloc(sizeof(double)* O);
	W = (double*)malloc(sizeof(double)* N * (M+1));
	W2 = (double*)malloc(sizeof(double)* N * (M + 1));
	check = (int*)malloc(sizeof(int)* N);
	index2 = (int*)malloc(sizeof(int)* N);
	A = (double*)malloc(sizeof(double)* M * M);
	num = (double*)malloc(sizeof(double)* 1);
	den = (double*)malloc(sizeof(double)* 1);
	P = (double*)malloc(sizeof(double)* N * N);
	errsum = (double*)malloc(sizeof(double)* N);

	for (i = 0; i < N; ++i) {
		check[i] = 0;
		index2[i] = -1;
	}

	for (i = 0; i < M; ++i) {
		itr = i * M;
		for (j = 0; j < M; ++j) {
			if (i == j) {
				A[itr + j] = 1.0;
			}
			else {
				A[itr + j] = 0.0;
			}
		}
	}
	actfcn_gauss2(data, N, I, data, N, beta, P);
	itr2 = 0;
	for (j = 0; j < M; ++j) {
		itr = j;
		w1t = 0;
		//actfcn_gauss6(data, N, I, data+j*I, beta, P);
		for (i = 0; i < N; ++i) {
			w1[i] = W[itr] = P[itr];
			w1t += w1[i] * w1[i];
			itr += N;
		}

		mmult(w1, target, g, 1, N, O);
		alpha = 1.0 / w1t;
		scale(g, 1, O, alpha);

		for (i = 0; i < O; ++i) {
			err[itr2 + i] = g[i] * g[i] * w1t / dd[i];
		}
		itr2 += O;
	}

	index = max_error(err, M, O,errsum);
	//mdisplay(err, M, O);
	check[index] = 1;
	itr = index;
	//actfcn_gauss6(data, N, I, data + itr*I, beta, P);
	//actfcn_gauss2(data, N, I, data,N, beta, P);
	for (i = 0; i < N; ++i) {
		W[i] = P[itr];
		itr += N;
	}

	for (i = 0; i < I; ++i) {
		centroid[i] = data[index * I + i];
	}

	tsvd = tloop = 0.0;
	icount = 1;
	for (k = 1; k < M; ++k) {
		count = 0;
		itr3 = 0;
		icount++;
		t0 = omp_get_wtime();
		for (i = 0; i < N; ++i) {
			if (check[i] == 0) {
				index2[count] = i;
				itr = i;
				//actfcn_gauss6(data, N, I, data + i*I, beta, P);
				for (j = 0; j < N; ++j) {
					w1[j] = P[itr];
					itr += N;
				}
				count++;
				itr = 0;
				itr2 = k;
				for (j = 0; j < k; ++j) {
					itr = j * N;
					mmult(W + itr, w1, num, 1, N, 1);
					mmult(W + itr, W + itr, den, 1, N, 1);
					A[itr2] = num[0] / den[0];
					itr2 += M;
				}
				itr = 0;
				itr2 = 0;

				itr = i;
				for (l = 0; l < N; ++l) {
					w1[l] = P[itr];
					itr += N;
				}
				itr2 = k;
				for (j = 0; j < k; ++j) {
					itr = j * N;
					for (l = 0; l < N; ++l) {
						w1[l] -= (A[itr2] * W[itr + l]);
					}
					itr2 += M;
				}
				mmult(w1, target, g, 1, N, O);
				mmult(w1, w1, &w1t, 1, N, 1);
				alpha = 1.0 / w1t;
				scale(g, 1, O, alpha);

				for (l = 0; l < O; ++l) {
					err[itr3 + l] = g[l] * g[l] * w1t / dd[l];
				}
				itr3 += O;
			}
		}

		t1 = omp_get_wtime();

		tloop += (t1 - t0);

		ind = max_error(err, count, O,errsum);
		index = index2[ind];
		check[index] = 1;
		itr3 = index;
		itr = k * N;
		//actfcn_gauss6(data, N, I, data + index*I, beta, P);
		for (l = 0; l < N; ++l) {
			W[itr + l] = P[itr3];
			itr3 += N;
		}
		itr4 = k * I;
		for (i = 0; i < I; ++i) {
			centroid[itr4+i] = data[index * I + i];
		}
		itr2 = k;
		for (j = 0; j < k; ++j) {
			itr3 = j * N;
			for (l = 0; l < N; ++l) {
				W[itr + l] -= (A[itr2] * W[itr3 + l]);
				//printf("%d %d %d %d %d \n",j,l, itr + l, itr2, itr3 + l);
			}
			itr2 += M;
		}
		//if (icount == scheck) {
			clusters = k + 1;

			if (clusters > M) {
				clusters = M;
			}
			actfcn_gauss3(data, N, I, centroid, clusters, beta, W2);
			K1 = clusters + 1;
			t0 = omp_get_wtime();
			if (N >= K1) {
				rnk = lls_svd_p(W2, target, N, K1, O, weight);
			}
			else {
				rnk = lls_svd_p_transpose(W2, target, N, K1, O, weight);
			}
			t1 = omp_get_wtime();

			tsvd += (t1 - t0);

			if (rnk == -1) {
				printf("SVD Computation Fails \n");
				exit(-1);
			}
			else {
				itranspose(weight, K1, O);
			}
			lwt = O * K1;
			orbf_sim(N, data, I, O, clusters, beta, lwt, weight, centroid, output);

			ssqr = 0.0;

			for (j = 0; j < O*N; ++j) {
				ssqr += ((output[j] - target[j]) * (output[j] - target[j]));
			}
			mse = ssqr / (double)(O*N);
			printf("Centroid %d MSE %g \n", clusters, mse);
			if (mse <= stol) {
				break;
			}
			//icount = 0;
		//}

	}

	clusters = k + 1;

	if (clusters > M) {
		clusters = M;
	}


	actfcn_gauss3(data, N, I, centroid, clusters, beta, W);
	K1 = clusters + 1;

	if (N >= K1) {
		rnk = lls_svd_p(W, target, N, K1, O, weight);
	}
	else {
		rnk = lls_svd_p_transpose(W, target, N, K1, O, weight);
	}

	//mdisplay(weight, K1, O);

	if (rnk == -1) {
		printf("SVD Computation Fails \n");
		exit(-1);
	}
	else {
		itranspose(weight, K1, O);
	}

	printf("TLOOP %g TSVD %g", tloop, tsvd);

	free(dd);
	free(err);
	free(w1);
	free(g);
	free(check);
	free(A);
	free(num);
	free(den);
	free(P);
	free(index2);
	free(errsum);
	free(W2);
	free(output);
	return clusters;
}

static int ols_opt(double *data, int N, int I, int M, int O, double beta, double stol, int scheck, double *weight, double *centroid, double *target) {
	int i, j, k, l, itr, itr2, itr3, itr4, index, m2, count, clusters, K1, rnk, ind, icount, lwt,itrp;
	int *check, *index2;
	double *err, *W, *w1, *g, *dt, *dtd, *dd, *A, *num, *den, *P, *errsum, *W2, *output;
	double w1t, alpha, ssqr, mse;
	double tsvd, tloop, t0, t1;

	dt = (double*)malloc(sizeof(double)* O * N);
	dtd = (double*)malloc(sizeof(double)* O * O);

	dd = (double*)malloc(sizeof(double)* O);

	mtranspose(target, N, O, dt);

	mmult(dt, target, dtd, O, N, O);

	free(dt);

	for (i = 0; i < O; ++i) {
		dd[i] = 0.0;
	}
	itr = 0;
	for (i = 0; i < O; ++i) {
		for (j = 0; j < O; ++j) {
			dd[j] += dtd[itr + j];
		}
		itr += O;
	}

	for (i = 0; i < O; ++i) {
		if (dd[i] == 0) {
			dd[i] = DBL_MAX;
		}
	}

	free(dtd);
	itr = 0;
	m2 = (M * (M - 1) / 2);
	err = (double*)malloc(sizeof(double)* N * O);
	output = (double*)malloc(sizeof(double)* N * O);
	w1 = (double*)malloc(sizeof(double)* N);
	g = (double*)malloc(sizeof(double)* O);
	W = (double*)malloc(sizeof(double)* N * (M + 1));
	W2 = (double*)malloc(sizeof(double)* N * (M + 1));
	check = (int*)malloc(sizeof(int)* N);
	index2 = (int*)malloc(sizeof(int)* N);
	A = (double*)malloc(sizeof(double)* M * M);
	num = (double*)malloc(sizeof(double)* 1);
	den = (double*)malloc(sizeof(double)* 1);
	P = (double*)malloc(sizeof(double)* N * N);
	errsum = (double*)malloc(sizeof(double)* N);

	for (i = 0; i < N; ++i) {
		check[i] = 0;
		index2[i] = -1;
	}

	for (i = 0; i < M; ++i) {
		itr = i * M;
		for (j = 0; j < M; ++j) {
			if (i == j) {
				A[itr + j] = 1.0;
			}
			else {
				A[itr + j] = 0.0;
			}
		}
	}
	actfcn_gauss2(data, N, I, data, N, beta, P);
	itranspose(P, N, N);
	itr2 = 0;
	for (j = 0; j < N; ++j) {
		itr = j * N;
		w1t = 0;
		//actfcn_gauss6(data, N, I, data+j*I, beta, P);
		for (i = 0; i < N; ++i) {
			w1t += P[itr+i] * P[itr+i];
		}

		mmult(P+itr, target, g, 1, N, O);
		alpha = 1.0 / w1t;
		scale(g, 1, O, alpha);

		for (i = 0; i < O; ++i) {
			err[itr2 + i] = g[i] * g[i] * w1t / dd[i];
		}
		itr2 += O;
	}

	index = max_error(err, M, O, errsum);
	//mdisplay(err, M, O);
	check[index] = 1;
	itr = index*N;
	//actfcn_gauss6(data, N, I, data + itr*I, beta, P);
	//actfcn_gauss2(data, N, I, data,N, beta, P);
	for (i = 0; i < N; ++i) {
		W[i] = P[itr + i];
	}

	for (i = 0; i < I; ++i) {
		centroid[i] = data[index * I + i];
	}

	tsvd = tloop = 0.0;
	icount = 1;
	for (k = 1; k < M; ++k) {
		count = 0;
		itr3 = 0;
		icount++;
		t0 = omp_get_wtime();
		for (i = 0; i < N; ++i) {
			if (check[i] == 0) {
				index2[count] = i;
				itr = i;
				itrp = i * N;
				//actfcn_gauss6(data, N, I, data + i*I, beta, P);
				//for (j = 0; j < N; ++j) {
					//w1[j] = P[itr];
					//itr += N;
				//}
				count++;
				itr = 0;
				itr2 = k;
				for (j = 0; j < k; ++j) {
					itr = j * N;
					mmult(W + itr, P+itrp, num, 1, N, 1);
					mmult(W + itr, W + itr, den, 1, N, 1);
					A[itr2] = num[0] / den[0];
					itr2 += M;
				}
				itr = 0;
				itr2 = 0;

				itr = i*N;
				for (l = 0; l < N; ++l) {
					w1[l] = P[itr+l];
				}
				itr2 = k;
				for (j = 0; j < k; ++j) {
					itr = j * N;
					for (l = 0; l < N; ++l) {
						w1[l] -= (A[itr2] * W[itr + l]);
					}
					itr2 += M;
				}
				mmult(w1, target, g, 1, N, O);
				mmult(w1, w1, &w1t, 1, N, 1);
				alpha = 1.0 / w1t;
				scale(g, 1, O, alpha);

				for (l = 0; l < O; ++l) {
					err[itr3 + l] = g[l] * g[l] * w1t / dd[l];
				}
				itr3 += O;
			}
		}

		t1 = omp_get_wtime();

		tloop += (t1 - t0);

		ind = max_error(err, count, O, errsum);
		index = index2[ind];
		check[index] = 1;
		itr3 = index*N;
		itr = k * N;
		//actfcn_gauss6(data, N, I, data + index*I, beta, P);
		for (l = 0; l < N; ++l) {
			W[itr + l] = P[itr3+l];
		}
		itr4 = k * I;
		for (i = 0; i < I; ++i) {
			centroid[itr4 + i] = data[index * I + i];
		}
		itr2 = k;
		for (j = 0; j < k; ++j) {
			itr3 = j * N;
			for (l = 0; l < N; ++l) {
				W[itr + l] -= (A[itr2] * W[itr3 + l]);
				//printf("%d %d %d %d %d \n",j,l, itr + l, itr2, itr3 + l);
			}
			itr2 += M;
		}
		//if (icount == scheck) {
		clusters = k + 1;

		if (clusters > M) {
			clusters = M;
		}
		actfcn_gauss3(data, N, I, centroid, clusters, beta, W2);
		K1 = clusters + 1;
		t0 = omp_get_wtime();
		if (N >= K1) {
			rnk = lls_svd_p(W2, target, N, K1, O, weight);
		}
		else {
			rnk = lls_svd_p_transpose(W2, target, N, K1, O, weight);
		}
		t1 = omp_get_wtime();

		tsvd += (t1 - t0);

		if (rnk == -1) {
			printf("SVD Computation Fails \n");
			exit(-1);
		}
		else {
			itranspose(weight, K1, O);
		}
		lwt = O * K1;
		orbf_sim(N, data, I, O, clusters, beta, lwt, weight, centroid, output);

		ssqr = 0.0;

		for (j = 0; j < O*N; ++j) {
			ssqr += ((output[j] - target[j]) * (output[j] - target[j]));
		}
		mse = ssqr / (double)(O*N);
		printf("Centroid %d MSE %g \n", clusters, mse);
		if (mse <= stol) {
			break;
		}
		//icount = 0;
		//}

	}

	clusters = k + 1;

	if (clusters > M) {
		clusters = M;
	}


	actfcn_gauss3(data, N, I, centroid, clusters, beta, W);
	K1 = clusters + 1;

	if (N >= K1) {
		rnk = lls_svd_p(W, target, N, K1, O, weight);
	}
	else {
		rnk = lls_svd_p_transpose(W, target, N, K1, O, weight);
	}

	//mdisplay(weight, K1, O);

	if (rnk == -1) {
		printf("SVD Computation Fails \n");
		exit(-1);
	}
	else {
		itranspose(weight, K1, O);
	}

	printf("TLOOP %g TSVD %g", tloop, tsvd);

	free(dd);
	free(err);
	free(w1);
	free(g);
	free(check);
	free(A);
	free(num);
	free(den);
	free(P);
	free(index2);
	free(errsum);
	free(W2);
	free(output);
	return clusters;
}

void orbfnet_train(orbfnet_object obj, int size, double *data, double sigma, double *target) {
	int d, clusters, p, lw, cw,M,scheck,i;
	double beta,stol,mse;
	double *output;
	clusters = size;
	d = obj->I;
	p = obj->O;
	lw = obj->lw;
	M = obj->hmax;
	stol = obj->stol;
	scheck = obj->scheck;

	//A = (double*)malloc(sizeof(double)* size * size);
	//x = (double*)malloc(sizeof(double)* lw);

	output = (double*)malloc(sizeof(double)* size * p);

	beta = sqrt(-log(0.5))/sigma;

	obj->beta = beta;

	// Gaussian Activation

	//actfcn_gauss2(data, size, d, data, clusters, beta, A);

	clusters = ols_cgs(data, size, d, M, p, beta, stol,scheck, obj->weight,obj->centroid, target);

	obj->H = clusters;

	lw = (obj->H + 1) * obj->O;
	cw = obj->I * obj->H;

	obj->lw = lw;

	orbf_sim(size, data, d, p, clusters, beta, lw, obj->weight, obj->centroid, output);
	obj->mse = 0.0;
	for (i = 0; i < size*p; ++i) {
		obj->mse += ((output[i] - target[i]) * (output[i] - target[i]));
	}

	obj->mse /= (double)(size*p);

	free(output);

}

void rbfnet_free(rbfnet_object obj) {
	free(obj);
}

void erbfnet_free(erbfnet_object obj) {
	free(obj);
}

void orbfnet_free(orbfnet_object obj) {
	free(obj);
}