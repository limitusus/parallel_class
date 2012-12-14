/* 
 * clustering
 */

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <mpi.h>
#include "Random123/threefry.h"

#ifndef dim
#define dim 2
#endif

typedef struct {
  double p[dim];
} point;

static double rnd_double(long x) {
  threefry2x64_key_t key = {{0, 0}};
  threefry2x64_ctr_t ctr = {{0, x}};
  threefry2x64_ctr_t r = threefry2x64(ctr, key);
  const int64_t two_to_the_62 = ((int64_t)1)<<62;
  const double two_to_the_62d = (double)two_to_the_62;
  return (r.v[0] & (two_to_the_62 - 1)) / (double)two_to_the_62d;
}

static void point_print(FILE * wp, point a) {
  int d;
  for (d = 0; d < dim; d++) {
    fprintf(wp, " %f", a.p[d]);
  }
}

static void point_zero(point * a) {
  int d;
  for (d = 0; d < dim; d++) a->p[d] = 0.0;
}

/* c = a + b */
static void point_plus(point * a, point * b, point * c) {
  int d;
  for (d = 0; d < dim; d++) c->p[d] = a->p[d] + b->p[d];
}

/* c = a * k */
static void point_mul(point * a, double k, point * c) {
  int d;
  for (d = 0; d < dim; d++) c->p[d] = a->p[d] * k;
}

/* c = a / k */
static void point_div(point * a, double k, point * c) {
  assert(k >= 0);
  point_mul(a, 1.0/k, c);
}

static double point_distance2(point * a, point * b) {
  double x = 0.0;
  int i;
  for (i = 0; i < dim; i++) {
    x += (a->p[i] - b->p[i]) * (a->p[i] - b->p[i]);
  }
  return x;
}

typedef struct {
  long n_members;
  point sum;
  point center;
} cluster;

const double R = 1024.0 * 1024.0 * 1024.0;

double set_nearest(int K, point * p, int  * m, cluster clusters[K]) {
  int nearest_cluster = -1;
  double nearest_dist = 0.0;
  int k;
  for (k = 0; k < K; k++) {
    double dist = point_distance2(p, &clusters[k].center);
    if (nearest_cluster == -1 || dist < nearest_dist) {
      nearest_cluster = k;
      nearest_dist = dist;
    }
  }
  assert(nearest_cluster != -1);
  *m = nearest_cluster;
  double dl = 0.0;
  int d;
  for (d = 0; d < dim; d++) {
    double dx = p->p[d] - clusters[nearest_cluster].center.p[d];
    if (dx < 0.0) dx = -dx;
    dl += log((long)(R * dx) + 1);
  }
  return dl;
}

void update_centers(long n, int K, point p[n],
		    int membership[n], cluster clusters[K]) {
  int i, j;
  for (j = 0; j < K; j++) {
    clusters[j].n_members = 0;
    point_zero(&clusters[j].sum);
  }
  for (i = 0; i < n; i++) {
    int m = membership[i];
    point_plus(&clusters[m].sum, &p[i], &clusters[m].sum);
    clusters[m].n_members++;
  }

  /* temporary buffers */
  double tmp_sum[2];
  long tmp_n_members;
  for (j = 0; j < K; j++) {
    tmp_sum[0] = clusters[j].sum.p[0];
    tmp_sum[1] = clusters[j].sum.p[1];
    tmp_n_members = clusters[j].n_members;
    MPI_Allreduce(tmp_sum, clusters[j].sum.p, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&tmp_n_members, &clusters[j].n_members, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (clusters[j].n_members > 0) {
      point_div(&clusters[j].sum, clusters[j].n_members, &clusters[j].center);
    }
  }
}

double update_membership(long n, int K, point p[n],
			 int membership[n], cluster clusters[K]) {
  double dl = 0.0;
  int i;
  for (i = 0; i < n; i++) {
    int m;
    dl += set_nearest(K, &p[i], &m, clusters);
    if (membership[i] != m) {
      membership[i] = m;
    }
  }

  double sum_dl;
  MPI_Allreduce(&dl, &sum_dl, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  //printf("%d changed\n", changed);
  return sum_dl;
}

void dump_plot(long n, int K, point * p, 
	       int * membership, cluster * clusters, FILE * wp) {
  int k = 0;
  long i = 0, j = 1;

  /* 
               i         j
     |<=|<=|<=|?|x|x|x|x|?|?|?| ... | | | | | | | | 
  
   */
  fprintf(stderr, "dump_plot sorting begin\n");
  while (i < n) {
#if 0
    printf("---- i = %d j = %d k = %d ----\n", i, j, k);
    for (t = 0; t < n; t++) {
      printf("%d ", membership[t]);
    }
    printf("\n");
#endif
#if 0
    for (t = 0; t < i; t++) {
      assert(membership[t] <= k);
      if (t > 0) {
	assert(membership[t - 1] <= membership[t]);
      }
    }
    for (t = i + 1; t < j; t++) {
      assert(membership[t] != k);
    }
#endif
    if (membership[i] != k) {
      if (j <= i) j = i + 1;
      for (; j < n; j++) {
	if (membership[j] == k) break;
      }
      if (j < n) {
	int m = membership[i];
	membership[i] = membership[j];
	membership[j] = m;
	point q = p[i];
	p[i] = p[j];
	p[j] = q;
	i++;
	j++;
      } else {
	k++;
	j = i + 1;
      }
    } else {
      i++;
    }
  }
  fprintf(stderr, "dump_plot sorting end\n");

  fprintf(wp, "plot");
  /* one plot for each cluster, plus cluster centers */
  for (k = 0; k < K + 1; k++) {
    if (k == 0) fprintf(wp, " ");
    else fprintf(wp, ",");
    fprintf(wp, "'-'");
  }
  fprintf(wp, "\n");
  k = 0;
  for (i = 0; i < n; i += n / 1024) {
    while (k < membership[i]) {
      fprintf(wp, "e\n");
      k++;
    }
    point_print(wp, p[i]);
    fprintf(wp, "\n");
  }
  while (k < K) {
    fprintf(wp, "e\n");
    k++;
  }
  for (k = 0; k < K; k++) {
    point_print(wp, clusters[k].center);
    fprintf(wp, "\n");
  }
  fprintf(wp, "e\n");
  fprintf(wp, "pause -1\n");
}

double kmeans1(long n, int K, 
	       point * p, 	/* n */
	       int * membership, /* n */
	       cluster * clusters, /* K */
	       int try,
               int rank) {
  int i;
  assert(K <= n);
  for (i = 0; i < n; i++) {
    membership[i] = 0;
  }
  int key = 200000 * K + try;

  for (i = 0; i < K; i++) {
    int d;
    for (d = 0; d < dim; d++) {
      double x;
      if (rank == 0) {
        x = rnd_double(key++);
      }
      MPI_Bcast(&x, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      clusters[i].center.p[d] = x;
    }
  }

  double dl = update_membership(n, K, p, membership, clusters);
  double dl_new = dl;
  int t;
  for (t = 0; t < 100; t++) {
    update_centers(n, K, p, membership, clusters);
    dl_new = update_membership(n, K, p, membership, clusters);
    if (rank == 0) {
    fprintf(stderr, 
	    "kmeans1(%lu, %d, try=%d) t=%d %f\n",
	    n, K, try, t, dl_new);
    }
    if (dl_new - dl > n * log(0.999)/log(2.0)) break;
    dl = dl_new;
  }
  //canonicalize(n, K, membership);
  return dl_new;
}

void swap01(int ** membership, cluster ** clusters, int a, int b) {
  int * m = membership[a];
  cluster * c = clusters[a];
  membership[a] = membership[b];
  clusters[a] = clusters[b];
  membership[b] = m;
  clusters[b] = c;
}


double kmeans(long n, int K, 
	      point * p, 
	      int ** membership, 
	      cluster ** clusters,
              int rank) {
  double min_dl = 0;
  int min_try = -1;
  int try;

  long all_n;
  MPI_Allreduce(&n, &all_n, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

  for (try = 0; try < 5; try++) {
    double dl = kmeans1(n, K, p, membership[1], clusters[1], try, rank);
    if (min_try == -1 || dl < min_dl) {
      /* keep the best value at membership[0] */
      swap01(membership, clusters, 0, 1);
      min_dl = dl;
      min_try = try;
    }
  }
  return log(R) * dim * K + all_n * log(K) + min_dl;
}

int main(int argc, char ** argv) {
  /* mpi initialization */
  int nprocs, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  double stime = MPI_Wtime();
  MPI_Status status;

  long i, j;

  long n; int max_K;
  int dimension;
  FILE * fp;
  point * p;

  long mystart, mynum;

  if (rank == 0) {
    char * filename = (argc > 1 ? argv[1] : "points.txt");
#if 0  
    int problem_no = (argc > 1 ? atoi(argv[1]) : 0);
    problem_clustering_t a = open_problem_clustering(problem_no);
#endif
    fp = fopen(filename, "rb");
    int rs = fscanf(fp, "%lu %d\n", &n, &dimension);
    assert(rs == 2);
    assert(dimension == 2);
    max_K = pow(n, 1.0/4.0);  
    MPI_Bcast(&n, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dimension, 1, MPI_INT, 0, MPI_COMM_WORLD);

    mystart = (n / nprocs) * rank;
    mynum = n / nprocs;

    double input_bytes = sizeof(point) * (double)n;
    double total_bytes = 3 * ((sizeof(point) + sizeof(int)) * (double)n 
                              + sizeof(cluster) * max_K);
    fprintf(stderr, 
            "%ld points, <= %d clusters, %.3f MB input %.3f MB total\n", 
            n, max_K, input_bytes * 1.0e-6, total_bytes * 1.0e-6);
    p = malloc(sizeof(point) * (n/nprocs));

    for (i = 0; i < mynum; i++) {
      rs = fscanf(fp, "%lf %lf\n", &p[i].p[0], &p[i].p[1]);
      assert(rs == 2);
    }
    double * buffer = malloc(sizeof(double) * mynum * 2);
    for (i = 1; i < nprocs; i++) {
      for (j = 0; j < mynum; j++) {
        rs = fscanf(fp, "%lf %lf\n", &buffer[2*j], &buffer[2*j+1]);
        assert(rs == 2);
      }
      MPI_Send(buffer, mynum * 2, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
    }
    free(buffer);
    fclose(fp);
  }
  else {
    MPI_Bcast(&n, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dimension, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("max_K recieved %d (rank = %d)\n", max_K, rank);

    mystart = (n / nprocs) * rank;
    mynum = n / nprocs;

    p = malloc(sizeof(point) * mynum);

    double * buffer = malloc(sizeof(double) * mynum * 2);
    MPI_Recv(buffer, mynum * 2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
    for (i = 0; i < mynum; i++) {
      p[i].p[0] = buffer[2*i];
      p[i].p[1] = buffer[2*i+1];
    }
    free(buffer);
  }

  int * membership[3] = { malloc(sizeof(int) * mynum), 
			  malloc(sizeof(int) * mynum),
			  malloc(sizeof(int) * mynum) };
  cluster * clusters[3] = { malloc(sizeof(cluster) * max_K), 
			    malloc(sizeof(cluster) * max_K),
			    malloc(sizeof(cluster) * max_K) };

  int K;
  double min_dl = 0.0;
  int min_K = -1;
  for (K = max_K; K >= 2; K = (int)(K * 2.0/3.0)) {
    if (rank == 0) {
      fprintf(stderr, "clustering to %d clusters\n", K);
    }
    double dl = kmeans(mynum, K, p, membership+1, clusters+1, rank);
    if (rank == 0) {
      fprintf(stderr, "  description length = %f\n", dl);
    }
    if (min_K == -1 || dl < min_dl) {
      /* keep the best value at membership[0] */
      swap01(membership, clusters, 0, 1);
      min_dl = dl;
      min_K = K;
    }
  }

  if (rank == 0) {
    fprintf(stderr, 
            "answer: %d clusters, description length = %f\n", 
            min_K, min_dl);
    FILE * wp = fopen("result.gpl", "wb");
    dump_plot(mynum, min_K, p, membership[0], clusters[0], wp);
    fclose(wp);
  }

  for (i = 0; i < 3; i++) {
    free(membership[i]);
    free(clusters[i]);
  }

  double etime = MPI_Wtime();
  if (rank == 0) {
    printf("*** took %f sec ***\n", etime - stime);
  }
  MPI_Finalize();
  return 0;
}
