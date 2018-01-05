#include "netdata.h"

ndata_object ndata_init(int inputs, int outputs, int patterns) {
	ndata_object obj = NULL;
	int dlen,dip,dop;

	dip = inputs * patterns;
	dop = outputs * patterns;

	dlen = dip + dop;

	obj = (ndata_object)malloc(sizeof(struct ndata_set) + sizeof(double)* dlen);

	obj->I = inputs;
	obj->O = outputs;
	obj->P = patterns;

	obj->tsize = (int) (0.7 * obj->P);
	obj->gsize = (int) (0.15 * obj->P);
	obj->vsize = obj->P - obj->tsize - obj->gsize;

	obj->data = &obj->params[0];
	obj->target = &obj->params[dip];

	return obj;
}

void interleave(double *inp, int size, int M, double *oup) {
	mtranspose(inp, M, size, oup);
}

void data_enter(ndata_object obj,double *data, double *target) {
	interleave(data, obj->P, obj->I, obj->data);
	interleave(target, obj->P, obj->O, obj->target);
}

void data_interleave_enter(ndata_object obj, double *data, double *target) {
	int i;

	for (i = 0; i < obj->P * obj->I; ++i) {
		obj->data[i] = data[i];
	}

	for (i = 0; i < obj->P * obj->O; ++i) {
		obj->target[i] = target[i];
	}
}

void csvreader(ndata_object obj,const char *filepath, const char *delimiter, int isHeader) {
	FILE *f;
	char buf[BUFFER];
	char *c;
	char *head;
	int i, j,len,iter,t1,t2;
	float *temp;

	temp = (float*)malloc(sizeof(float)* (obj->I + obj->O));

	f = fopen(filepath, "r");
	i = t1 = t2 = 0;
	len = obj->I + obj->O;
	if (f == NULL) {
		printf("Error Opening File \n");
		exit(1);
	}
	else {
		if (isHeader != 0) {
			fgets(buf, sizeof(buf), f);
			c = strtok(buf, delimiter);
			while (c != NULL) {
				head = c; // iterate through the header. Don't save it for now.
				c = strtok(NULL, delimiter);
			}
		}
		while (fgets(buf, sizeof(buf), f) && i < obj->P) {
			j = 0;
			c = strtok(buf, delimiter);
			while (c != NULL && j < len) {
				temp[j] = atof(c);
				c = strtok(NULL, delimiter);
				j++;
			}
			i++;
			for (iter = 0; iter < obj->I; iter++) {
				obj->data[t1+iter] = (double) temp[iter];
			}
			for (iter = obj->I; iter < len; iter++) {
				obj->target[t2+iter - obj->I] = (double) temp[iter];
			}

			t1 += obj->I;
			t2 += obj->O;
		}
	}

	fclose(f);
	free(temp);
}

void file_sep_line_enter(ndata_object obj, const char *filepath, const char *delimiter, int isHeader) {
	FILE *f;
	char buf[BUFFER];
	char *c;
	char *head;
	int i, p2,j, len, iter, t1, t2;
	float *temp;

	temp = (float*)malloc(sizeof(float)* (obj->I + obj->O));

	f = fopen(filepath, "r");
	i = t1 = t2 = 0;
	len = obj->I + obj->O;
	p2 = 2 * obj->P;
	if (f == NULL) {
		printf("Error Opening File \n");
		exit(1);
	}
	else {
		if (isHeader != 0) {
			fgets(buf, sizeof(buf), f);
			c = strtok(buf, delimiter);
			while (c != NULL) {
				head = c; // iterate through the header. Don't save it for now.
				c = strtok(NULL, delimiter);
			}
		}
		while (fgets(buf, sizeof(buf), f) && i < p2) {
			if (i % 2 == 0) {
				len = obj->I;
				j = 0;
				c = strtok(buf, delimiter);
				while (c != NULL && j < len) {
					temp[j] = atof(c);
					c = strtok(NULL, delimiter);
					j++;
				}
				i++;
				for (iter = 0; iter < len; iter++) {
					obj->data[t1 + iter] = (double)temp[iter];
				}
				t1 += obj->I;
			}
			else {
				len = obj->O;
				j = 0;
				c = strtok(buf, delimiter);
				while (c != NULL && j < len) {
					temp[j] = atof(c);
					c = strtok(NULL, delimiter);
					j++;
				}
				i++;
				for (iter = 0; iter < len; iter++) {
					obj->target[t2 + iter] = (double)temp[iter];
				}

				t2 += obj->O;
			}
			
		}
	}

	fclose(f);
	free(temp);
}

void file_enter(ndata_object obj, const char *filepath, const char *delimiter, int firstlineheader) {
	csvreader(obj, filepath, delimiter, firstlineheader);

}

void file_rev_enter(ndata_object obj, const char *filepath, const char *delimiter, int isHeader) {
	FILE *f;
	char buf[BUFFER];
	char *c;
	char *head;
	int i, j, len, iter, t1, t2;
	float *temp;

	temp = (float*)malloc(sizeof(float)* (obj->I + obj->O));

	f = fopen(filepath, "r");
	i = t1 = t2 = 0;
	len = obj->I + obj->O;
	if (f == NULL) {
		printf("Error Opening File \n");
		exit(1);
	}
	else {
		if (isHeader != 0) {
			fgets(buf, sizeof(buf), f);
			c = strtok(buf, delimiter);
			while (c != NULL) {
				head = c; // iterate through the header. Don't save it for now.
				c = strtok(NULL, delimiter);
			}
		}
		while (fgets(buf, sizeof(buf), f) && i < obj->P) {
			j = 0;
			c = strtok(buf, delimiter);
			while (c != NULL && j < len) {
				temp[j] = atof(c);
				c = strtok(NULL, delimiter);
				j++;
			}
			i++;
			for (iter = 0; iter < obj->O; iter++) {
				obj->target[t1 + iter] = (double)temp[iter];
			}
			for (iter = obj->O; iter < len; iter++) {
				obj->data[t2 + iter - obj->O] = (double)temp[iter];
			}

			t1 += obj->O;
			t2 += obj->I;
		}
	}

	fclose(f);
	free(temp);
}

void ndata_check(ndata_object obj) {
	int i,inp,oup,p;

	p = obj->P;
	inp = obj->I;
	oup = obj->O;
	printf("\n");
	printf("Number Of Patterns : \t%d\n", p);
	printf("Number Of Inputs   : \t%d\n", inp);
	printf("Number Of Outputs  : \t%d\n", oup);
	printf("\n\n");
	printf("INPUTS : \n");
	printf("Pattern %d :\t", 1);
	for (i = 0; i < inp; ++i) {
		printf("%g\t", obj->data[i]);
	}
	printf("\n");
	printf("Pattern %d :\t", p);
	for (i = (p-1)*inp; i < inp*p; ++i) {
		printf("%g\t", obj->data[i]);
	}
	printf("\n\n");
	printf("TARGETS : \n");
	printf("Pattern %d :\t", 1);
	for (i = 0; i < oup; ++i) {
		printf("%g\t", obj->target[i]);
	}
	printf("\n");
	printf("Pattern %d :\t", p);
	for (i = (p - 1)*oup; i < oup*p; ++i) {
		printf("%g\t", obj->target[i]);
	}
	printf("\n");
}

void ndata_free(ndata_object obj) {
	free(obj);
}