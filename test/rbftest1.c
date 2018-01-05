#include <stdio.h>
#include <stdlib.h>
#include "../header/rbf.h"

int main() {
    ndata_object data;
	rbfnet_object net;

	int inp, oup, patterns;
	double *ys;
	int hidden;
	int tsize, itrd, itrt, i, j, t1, t2;
	int isheader = 0;
	char* file = "iris.data.txt";
	char *delimiter = " ";

	inp = 4;
	oup = 3;
	patterns = 150;
	hidden = 40;
	tsize = 50;
	itrd = tsize * inp;
	itrt = tsize * oup;

	ys = (double*)malloc(sizeof(double)* 20 * oup);

	data = ndata_init(inp, oup, patterns);
	file_enter(data, file, delimiter, isheader);
	ndata_check(data);

	net = rbfnet_init(inp, oup, hidden);

	rbfnet_set_training_ratios(net, 0.8, 0.2);

	rbfnet_train(net, patterns, data->data, data->target);

	rbfnet_sim(net, 20, data->data + itrd, ys);

	for (i = 0; i < 20; ++i) {
		t1 = oup * i;
		t2 = oup * (i + tsize);
		for (j = 0; j < oup; ++j) {
			printf("%g %g  ", data->target[t2 + j], ys[t1 + j]);
		}
		printf("\n");
	}

	ndata_free(data);
	rbfnet_free(net);

	free(ys);
    return 0;
}