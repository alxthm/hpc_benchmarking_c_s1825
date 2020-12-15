#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <immintrin.h>
#include <pthread.h>

#define MODE_NORM_D 0
#define MODE_NORM_NAIVE 1
#define MODE_NORM_REC 2
#define MODE_NORM_VEC 3

#define MAX_NB_THREADS 64


// --- utils
double now() {
  // get current time in milliseconds
  struct timeval t; double f_t;
  gettimeofday(&t, NULL);
  f_t = t.tv_usec / 1000.;
  f_t += t.tv_sec * 1000.;
  return f_t;
}

char* get_mode_str(int mode) {
  if (mode == MODE_NORM_D)
    return "double";
  else if (mode == MODE_NORM_NAIVE)
    return "naive";
  else if (mode == MODE_NORM_REC)
    return "rec";
  else if (mode == MODE_NORM_VEC)
    return "vec";
  else
    return "_";
}

int write_to_file(float* U, int n) {
  // Format the file name depending on the n
  char* file_name;
  if(0 > asprintf(&file_name, "array_%d.csv", n))
    return -1;

  // Write to the file
  FILE* f = fopen(file_name, "w");
  for (int i = 0; i < n; i++)
    fprintf(f, "%f\n", U[i]);

  free(file_name);
  fclose(f);
  return 0;
}

// --- norm functions
float norm_naive(float* U, int n) {
  // Naive sum
  float sum = 0.f;
  for (int i = 0; i < n; i++) {
    sum += sqrtf(fabs(U[i]));
  }
  return sum;
}

float norm_d(float* U, int n) {
  // Use double precision for the sum (in order to limit numerical errors with larger n)
  double sum = 0.f;
  for (int i = 0; i < n; i++) {
    sum += sqrtf(fabs(U[i]));
  }
  return (float) sum;
}

float norm_recursive(float* U, int n) {
  // Recursively add digits so they are always on the same magnitude,
  // in order to limit numerical errors (but still using floats)
  if (n == 1) {
    return sqrtf(fabs(U[0]));
  }
  else {
    int n1 = n / 2;
    int n2 = n - n1;
    return norm_recursive(&U[0], n1) + norm_recursive(&U[n1], n2);
  }
}

double norm_oracle(float* U, int n) {
  // Return the result as a double (very close to the true value given our problem)
  double sum = 0.f;
  for (int i = 0; i < n; i++) {
    sum += sqrtf(fabs(U[i]));
  }
  return sum;
}

// Note (Q5) :
// - we can work with non-aligned data by using the _mm256_loadu_ps method
//  instead of _mm256_load_ps
// - in order to work with arbitrary values of n, we could easily do the same
//  work up to i=8*(n/8), and sum the remaining values like in the naive function
float norm_vec(float* U, int n) {
  // number of operations (assume n is a multiple of 8)
  int m = n / 8;

  // Alternative approach to our signmask:
  // __m256i minus1_256 = _mm256_set1_epi32(-1);
  // __m256 absmask_256 = _mm256_castsi256_ps(_mm256_srli_epi32(minus1_256, 1));
  // u = _mm256_and_ps(absmask_256, u);

  __m256 sum = _mm256_set1_ps(0.f);
  __m256 signmask = _mm256_set1_ps(-0.f); // sign bit to 1, all others to 0
    for (int i = 0; i < m; i++) {
    __m256 u = _mm256_load_ps(&U[8*i]);
    // this changes the sign bit of u to 0, i.e. makes it a positive number
    // while keeping the others to the original value
    u = _mm256_andnot_ps(signmask, u);
    u = _mm256_sqrt_ps(u);
    sum = _mm256_add_ps(sum, u);
  }

  float total_sum = 0.f;
  for (int i = 0; i < 8; i++) {
    total_sum += sum[i];
  }
  return total_sum;
}

float norm(float* U, int n, int mode) {
  float sum = 0.f;
  if (mode == MODE_NORM_NAIVE) {
    sum = norm_naive(U, n);
  }
  else if (mode == MODE_NORM_D) {
    sum = norm_d(U, n);
  }
  else if (mode == MODE_NORM_REC) {
    sum = norm_recursive(U, n);
  }
  else if (mode == MODE_NORM_VEC) {
    sum = norm_vec(U, n);
  }
  return sum;
}


// --- multi-threading code
float sum_results[MAX_NB_THREADS]; // array containing the results of each thread
// note: we could also have used a simple float variable with a mutex,
// or pass the sum_results array as a parameter through thread_data, if we
// wanted to avoid using a global variable

struct thread_data {
  unsigned int thread_id;
  int mode;
  float* V;  // sub array on which the thread should operate
  int m;  // length of the sub array
};

void* thread_function(void* thread_arg) {
  // Access shared data
  struct thread_data* data = (struct thread_data*)thread_arg;
  int thread_id = data->thread_id;
  int mode = data->mode;
  float* V = data->V;
  int m = data->m;

  // Do the computation
  float sum = norm(V, m, mode);

  // store the results in a shared global variable
  sum_results[thread_id] = sum;

  pthread_exit(NULL);
}

float norm_par(float *U, int n, int mode, int nb_threads) {
  struct thread_data thread_data_array[nb_threads];
  pthread_t thread_array[nb_threads];
  int rc;

  // Compute the results in parallel
  for (int i = 0; i < nb_threads; i++) {
    // assume n is a multiple of nb_threads * 8
    thread_data_array[i].thread_id = i;
    thread_data_array[i].mode = mode;
    thread_data_array[i].V = &U[i * (n / nb_threads)];
    thread_data_array[i].m = n / nb_threads;
    rc = pthread_create(&thread_array[i], NULL, thread_function,
      &thread_data_array[i]);
    if (rc){
      printf("ERROR; return code from pthread_create() (id %d) is %d\n", i, rc);
      exit(-1);
    }
  }

  // Wait for all threads to finish
  for (int i = 0; i < nb_threads; i++) {
    rc = pthread_join(thread_array[i], NULL);
    if (rc) {
      printf("ERROR; return code from pthread_join() (id %d) is %d\n", i, rc);
      exit(-1);
    }
  }

  // Sum all results and return the final sum
  float sum = 0.;
  for (int i = 0; i < nb_threads; i++) {
    sum += sum_results[i];
  }
  return sum;
}


// ---- main
int run_quick_test(float* U, int n, int nb_threads);
int run_trials(float* U, int n);


// To have a quick recap with n=256000 and nb_threads=4 (nb_threads = -1 means
// no multithreading), run :
// $ ./project 256000 4
// To run several trials with n=256000 (and various nb of threads), run :
// $ ./project 256000 1 1
int main(int argc, char const **argv) {
  srand(270);

  // --- Get commandline args
  // Size of the data n (256000 by default)
  int n = 256000;
  // nb_threads (only used in run_quick_test mode)
  int nb_threads = 1;
  // run a quick test (by default) or run comprehensive trials
  int should_run_trials = 0;
  if (argc > 1) n=atoi(argv[1]);
  if (argc > 2) nb_threads=atoi(argv[2]);
  if (argc > 3) should_run_trials=atoi(argv[3]);

  // --- Create data
  // create random vector U of size n (on the heap, as the stack is
  // not enough for large n)
  float* U = (float*) aligned_alloc(32, n * sizeof(float));
  // Note: in our experiments, aligning the array did not speed up the
  // vectorized function
  // float* U = (float*) malloc(n * sizeof(float));
  for (int i = 0; i < n; i++)
    // random float between -1 and 1
    U[i] = (2 * (float)rand() / RAND_MAX) - 1.f;

  // Save U so we can run experiments in python on the same data
  // write_to_file(U, n);

  if (should_run_trials)
    run_trials(U, n);
  else
    run_quick_test(U, n, nb_threads);

  free(U);
  return 0;
}

int run_quick_test(float* U, int n, int nb_threads) {
  double t0, t1;

  // ex.1 - naive for loop
  t0 = now();
  float s = norm_naive(U, n);
  t1 = now();
  double dt_naive = t1 - t0;

  t0 = now();
  double s_d = norm_d(U, n);
  t1 = now();
  double dt_d = t1 - t0;

  t0 = now();
  float s_rec = norm_recursive(U, n);
  t1 = now();
  double dt_rec = t1 - t0;

  t0 = now();
  double s_oracle = norm_oracle(U, n);
  t1 = now();
  double dt_oracle = t1 - t0;

  // ex.2 - vectorized
  t0 = now();
  float s_vec = norm_vec(U, n);
  t1 = now();
  double dt_vec = t1 - t0;

  // ex.3 - multi-threaded
  t0 = now();
  float s_par_d = norm_par(U, n, MODE_NORM_D, nb_threads);
  t1 = now();
  double dt_par_d = t1 - t0;

  t0 = now();
  float s_par_vec = norm_par(U, n, MODE_NORM_VEC, nb_threads);
  t1 = now();
  double dt_par_vec = t1 - t0;

  t0 = now();
  float s_par_rec = norm_par(U, n, MODE_NORM_REC, nb_threads);
  t1 = now();
  double dt_par_rec = t1 - t0;

  printf("oracle      : %f (%.5f ms)\n", s_oracle, dt_oracle);
  printf("norm        : %f (%.5f ms)\n", s, dt_naive);
  printf("norm_d      : %f (%.5f ms)\n", s_d, dt_d);
  printf("norm_rec    : %f (%.5f ms)\n", s_rec, dt_rec);
  printf("norm_vec    : %f (%.5f ms)\n", s_vec, dt_vec);
  printf("norm_par_d  : %f (%.5f ms)\n", s_par_d, dt_par_d);
  printf("norm_par_rec: %f (%.5f ms)\n", s_par_rec, dt_par_rec);
  printf("norm_par_vec: %f (%.5f ms)\n", s_par_vec, dt_par_vec);

  return 0;
}

int run_trials(float* U, int n) {
  double t0, t1, dt;

  printf("n,nb_threads,mode,trial,sum,time_ms\n");
  double s_oracle = norm_oracle(U, n);
  printf("%d,%d,oracle,%d,%f,%f\n", n, -1, 0, s_oracle, 0.);

  int nb_threads_to_try[7] = {-1, 1, 2, 4, 8, 16, 32};
  for (int i = 0; i < 7; i++) {
    // Compare runs with different number of threads
    int nb_threads = nb_threads_to_try[i];
    for (int mode = 0; mode < 4; mode++) {
      // Compare runs with different norm functions
      char* mode_str = get_mode_str(mode);
      for (int j = 0; j < 5; j++) {
        // Do 5 runs of each to get a better estimate of the time
        float s;
        if (nb_threads < 0) {
          t0 = now();
          s = norm(U, n, mode);
          t1 = now();
          dt = t1 - t0;
        }
        else {
          t0 = now();
          s = norm_par(U, n, mode, nb_threads);
          t1 = now();
          dt = t1 - t0;
        }
        printf("%d,%d,%s,%d,%f,%f\n", n, nb_threads, mode_str, j, s, dt);
      }
    }
  }
  return 0;
}
