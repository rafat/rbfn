#include <stdio.h>
#include <stdlib.h>
#include "../header/rbf.h"

int main() {
    ndata_object data;
	erbfnet_object net;

	int inp, oup, patterns;
	double *ys;
	int tsize, itrd, itrt, i, j, t1, t2;
	int isheader = 0;
	double sigma;
	char* file = "iris.data.txt";
	char *delimiter = " ";

	inp = 4;
	oup = 3;
	patterns = 150;
	tsize = 60;
	itrd = tsize * inp;
	itrt = tsize * oup;

	ys = (double*)malloc(sizeof(double)* 20 * oup);

	data = ndata_init(inp, oup, patterns);
	file_enter(data, file, delimiter, isheader);
	ndata_check(data);

	net = erbfnet_init(inp, oup, patterns);

	//rbfnet_set_training_ratios(net, 0.8, 0.2);
	sigma = 1.0;
	erbfnet_train(net, patterns, data->data, sigma, data->target);

	erbfnet_sim(net, 20, data->data + itrd, ys);

	for (i = 0; i < 20; ++i) {
		t1 = oup * i;
		t2 = oup * (i + tsize);
		for (j = 0; j < oup; ++j) {
			printf("%g %g  ", data->target[t2 + j], ys[t1 + j]);
		}
		printf("\n");
	}

	ndata_free(data);
	erbfnet_free(net);

	free(ys);
    return 0;
}