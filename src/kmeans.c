#include "kmeans.h"

double l2s(double *val1,double *val2, int N) {
	double l2norm,temp;
	int i;

	l2norm = 0.0;

	for (i = 0; i < N; ++i) {
		temp = val1[i] - val2[i];
		l2norm += (temp * temp);
	}

	return l2norm;
}

double l2(double *val1, double *val2, int N) {
	double l2norm;

	l2norm = l2s(val1, val2, N);
	l2norm = sqrt(l2norm);

	return l2norm;
}

static int minInd(double *inp, int d) {
	int index,i;
	double minval;

	minval = inp[0];
	index = 0;

	for (i = 1; i < d; ++i) {
		if (inp[i] < minval) {
			minval = inp[i];
			index = i;
		}
	}

	return index;
}

static int roulette(int N,double *prob,double sum) {
	double cuprob,iprob,prb;
	int i,index;

	cuprob = 0.0;
	index = -1;
	prb = (double)rand() / ((double)RAND_MAX + 1);

	for (i = 0; i < N; ++i) {
		iprob = prob[i] / sum;
		cuprob += iprob;
		if (prb < cuprob) {
			index = i;
			break;
		}
	}

	return index;
}

static int checkindex(int ind, int *index,int N) {
	int i,retval;

	retval = 1;
	if (ind == -1) {
		return retval;
	}
	else {
		for (i = 0; i < N; ++i) {
			if (ind == index[i]) {
				return retval;
			}
		}
	}
	retval = 0;
	return retval;
}

void initkmeans(double *x,int N, int d, int clusters, double *centroid) {
	int ind1,i,j,k;
	int iterx,iterc,M,intr,cindex,citer;
	double psum;
	int *index;
	srand(time(NULL));
	ind1 = (int) ((double)rand() / ((double)RAND_MAX + 1) * N);

	double *dist,*d2min;

	dist = (double*)malloc(sizeof(double) * N);
	d2min = (double*)malloc(sizeof(double)* clusters);
	index = (int*)malloc(sizeof(int)* clusters);

	for (i = 0; i < d; ++i) {
		centroid[i] = x[d*ind1 + i];
	}
	index[0] = ind1;

	iterx = 0;
	cindex = 1;
	for (j = 1; j < clusters; ++j) {
		psum = 0.0;
		for (i = 0; i < N; ++i) {
			iterc = 0;
			for (k = 0; k < j; ++k) {
				d2min[k] = l2(x+iterx,centroid+iterc,d);
				iterc += d;
			}

			M = minInd(d2min, j);
			iterx += d;
			dist[i] = d2min[M] * d2min[M];
			psum += dist[i];
		}
		citer = 0;
		while (cindex == 1 && citer < 100) {
			intr = roulette(N,dist,psum);
			//printf("%d \n", intr);
			cindex = checkindex(intr, index, j);
			citer++;
		}
		if (citer == 100) {
			printf("Error kmeans couldn't be initialized.\n");
			exit(-1);
		}

		for (i = 0; i < d; ++i) {
			centroid[j*d+i] = x[d*intr + i];
		}
		index[j] = intr;
		cindex = 1;
		iterx = 0;
	}

	free(dist);
	free(d2min);
	free(index);
}

void initkmeans_rand(double *x, int N, int d, int clusters, double *centroid) {
	int ind1,i,j,cindex,citer;
	int *index;
	srand(time(NULL));
	index = (int*)malloc(sizeof(int)* clusters);
	ind1 = (int)((double)rand() / ((double)RAND_MAX + 1) * N);

	for (i = 0; i < d; ++i) {
		centroid[i] = x[d*ind1 + i];
	}
	index[0] = ind1;

	cindex = 1;
	citer = 0;
	for (i = 1; i < clusters; ++i) {
		while (cindex == 1) {
			ind1 = (int)((double)rand() / ((double)RAND_MAX + 1) * N);
			for (j = 0; j < i; ++j) {
				if (ind1 == index[j]) {
					citer++;
				}
			}
			if (citer == 0) {
				for (j = 0; j < d; ++j) {
					centroid[i*d+j] = x[d*ind1 + j];
				}
				index[i] = ind1;
				cindex = 0;
			}
		}

		cindex = 1;
	}

	free(index);
}

int kmeans(double *x, int N, int d, int clusters, double *centroid,int *clusterindex) {
	int i,j,k,l;
	int status;// Return flag, 0 - Convergence, 1 - One of the Cluster becomes zero, 2 - Max Iterations reached 
	double *distmat;
	int *index,*clarray;
	int Max_Iter,iterx,iterc,M,citer,czeroc;

	Max_Iter = N * 20;

	index = (int*)malloc(sizeof(int)* N);// Maps data items to cluster
	distmat = (double*)malloc(sizeof(double)* N * clusters);// Distance Matrix
	clarray = (int*)malloc(sizeof(int)* clusters);

	initkmeans(x, N, d, clusters, centroid);

	for (i = 0; i < N; ++i) {
		index[i] = -1;
	}
	status = 0;

	for (i = 0; i < Max_Iter; ++i) {
		iterx = 0;
		citer = 0;
		for (j = 0; j < N; ++j) {
			iterc = 0;
			for (k = 0; k < clusters; ++k) {
				distmat[j*clusters+k] = l2s(x+iterx, centroid+iterc, d);
				iterc += d;
			}
			M = minInd(distmat+j*clusters, clusters);
			if (M != index[j]) {
				index[j] = M;
				citer++;
			}
			iterx += d;
		}

		for (k = 0; k < clusters; ++k) {
			clarray[k] = 0;
		}

		for (j = 0; j < N; ++j) {
			clarray[index[j]] += 1;
		}

		czeroc = 0;

		for (k = 0; k < clusters; ++k) {
			if (clarray[k] == 0) {
				czeroc += 1;
			}
		}

		if (citer == 0 ) {
			break;
		}

		if (czeroc != 0) {
			status = 1;
			break;
		}
		// Accept new Cluster index

		for (j = 0; j < N; ++j) {
			clusterindex[j] = index[j];
		}

		// Update centroids

		//Init to zero
		for (j = 0; j < clusters; ++j) {

			for (k = 0; k < d; ++k) {
				centroid[j*d + k] = 0.0;
			}

		}

		for (j = 0; j < N; ++j) {
			l = index[j];
			for (k = 0; k < d; ++k) {
				centroid[l*d + k] += x[j*d+k];
			}
		}

		for (j = 0; j < clusters; ++j) {

			for (k = 0; k < d; ++k) {
				centroid[j*d + k] /= clarray[j];
			}

		}

	}

	if (i == Max_Iter) {
		status = 2;
	}


	free(index);
	free(distmat);
	free(clarray);
	return status;
}