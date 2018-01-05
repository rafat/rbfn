#ifndef KMEANS_H_
#define KMEANS_H_

#include <limits.h>
#include <time.h>
#include "netdata.h"

#ifdef __cplusplus
extern "C" {
#endif

double l2s(double *val1, double *val2, int N);

double l2(double *val1, double *val2, int N);

void initkmeans(double *x, int N, int d, int clusters, double *centroid);

void initkmeans_rand(double *x, int N, int d, int clusters, double *centroid);

int kmeans(double *x, int N, int d, int clusters, double *centroid, int *clusterindex);

#ifdef __cplusplus
}
#endif

#endif /* KMEANS_H_ */