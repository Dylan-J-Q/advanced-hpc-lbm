#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"
#define BLOCK_X         32
#define BLOCK_Y         8

#define CUDA_CHECK(call) do {                                           \
    cudaError_t _err = (call);                                          \
    if (_err != cudaSuccess) {                                          \
      fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
              __FILE__, __LINE__, cudaGetErrorString(_err));            \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
} while (0)

typedef struct {
  int   nx;
  int   ny;
  int   maxIters;
  int   reynolds_dim;
  float density;
  float accel;
  float omega;
} t_param;

typedef struct {
  float *speeds[NSPEEDS];
} t_speed;

int  initialise(const char *paramfile, const char *obstaclefile,
                t_param *params,
                t_speed *h_cells, int **h_obstacles_ptr,
                float **av_vels_ptr);

int  write_values(const t_param params, const t_speed *h_cells,
                  const int *h_obstacles, const float *av_vels);

float calc_reynolds(const t_param params,
                         const t_speed *h_cells, const int *h_obstacles);

void die(const char *message, const int line, const char *file);
void usage(const char *exe);

__global__ void accelerate_flow_kernel(
    float * __restrict__ c1, float * __restrict__ c3,
    float * __restrict__ c5, float * __restrict__ c6,
    float * __restrict__ c7, float * __restrict__ c8,
    const int * __restrict__ obs,
    int nx, int ny, float density, float accel);

__global__ void propagate_collide_kernel(
    const float * __restrict__ c0, const float * __restrict__ c1,
    const float * __restrict__ c2, const float * __restrict__ c3,
    const float * __restrict__ c4, const float * __restrict__ c5,
    const float * __restrict__ c6, const float * __restrict__ c7,
    const float * __restrict__ c8,
    float * __restrict__ out0, float * __restrict__ out1,
    float * __restrict__ out2, float * __restrict__ out3,
    float * __restrict__ out4, float * __restrict__ out5,
    float * __restrict__ out6, float * __restrict__ out7,
    float * __restrict__ out8,
    const int * __restrict__ obs,
    float * __restrict__ vel_out,
    int nx, int ny, float omega);

int main(int argc, char *argv[])
{
  if (argc != 3) usage(argv[0]);
  const char *paramfile    = argv[1];
  const char *obstaclefile = argv[2];

  t_param  params;
  t_speed  h_cells;
  int     *h_obstacles = NULL;
  float   *av_vels     = NULL;

  struct timeval timstr;
  gettimeofday(&timstr, NULL);
  double tot_tic  = timstr.tv_sec + timstr.tv_usec / 1e6;
  double init_tic = tot_tic;

  initialise(paramfile, obstaclefile, &params, &h_cells, &h_obstacles, &av_vels);

  const int nx      = params.nx;
  const int ny      = params.ny;
  const int n_cells = nx * ny;

  // count fluid cells
  int fluid_cells = 0;
  for (int i = 0; i < n_cells; i++)
    if (!h_obstacles[i]) fluid_cells++;
  const float inv_fluid_cells = (fluid_cells > 0) ? 1.f / (float)fluid_cells : 0.f;

  // allocate mem on gpu
  t_speed d_cells, d_tmp_cells;
  for (int k = 0; k < NSPEEDS; k++) {
    CUDA_CHECK(cudaMalloc(&d_cells.speeds[k],     n_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tmp_cells.speeds[k], n_cells * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_cells.speeds[k], h_cells.speeds[k],
                          n_cells * sizeof(float), cudaMemcpyHostToDevice));
  }

  int *d_obstacles = NULL;
  CUDA_CHECK(cudaMalloc(&d_obstacles, n_cells * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_obstacles, h_obstacles,
                        n_cells * sizeof(int), cudaMemcpyHostToDevice));

  float *d_vel = NULL;
  CUDA_CHECK(cudaMalloc(&d_vel, n_cells * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_vel, 0, n_cells * sizeof(float)));

  /* launch configuration */
  dim3 block2d(BLOCK_X, BLOCK_Y);
  dim3 grid2d((nx + BLOCK_X - 1) / BLOCK_X,
              (ny + BLOCK_Y - 1) / BLOCK_Y);

  const int accel_threads = 128;
  const int accel_blocks  = (nx + accel_threads - 1) / accel_threads;

  thrust::device_ptr<float> d_vel_ptr(d_vel);

  CUDA_CHECK(cudaDeviceSynchronize());

  gettimeofday(&timstr, NULL);
  double init_toc = timstr.tv_sec + timstr.tv_usec / 1e6;
  double comp_tic = init_toc;

  // timestep loop
  t_speed *cur = &d_cells;
  t_speed *nxt = &d_tmp_cells;

  for (int tt = 0; tt < params.maxIters; tt++)
  {
    accelerate_flow_kernel<<<accel_blocks, accel_threads>>>(
        cur->speeds[1], cur->speeds[3],
        cur->speeds[5], cur->speeds[6],
        cur->speeds[7], cur->speeds[8],
        d_obstacles, nx, ny, params.density, params.accel);

    propagate_collide_kernel<<<grid2d, block2d>>>(
        cur->speeds[0], cur->speeds[1], cur->speeds[2],
        cur->speeds[3], cur->speeds[4], cur->speeds[5],
        cur->speeds[6], cur->speeds[7], cur->speeds[8],
        nxt->speeds[0], nxt->speeds[1], nxt->speeds[2],
        nxt->speeds[3], nxt->speeds[4], nxt->speeds[5],
        nxt->speeds[6], nxt->speeds[7], nxt->speeds[8],
        d_obstacles, d_vel,
        nx, ny, params.omega);

    float tot_u = thrust::reduce(thrust::device,
                                 d_vel_ptr, d_vel_ptr + n_cells, 0.f);
    av_vels[tt] = tot_u * inv_fluid_cells;

    t_speed *tmp = cur; cur = nxt; nxt = tmp;
  }

  CUDA_CHECK(cudaDeviceSynchronize());

  gettimeofday(&timstr, NULL);
  double comp_toc = timstr.tv_sec + timstr.tv_usec / 1e6;
  double col_tic  = comp_toc;

  //send final state back to cpu
  for (int k = 0; k < NSPEEDS; k++)
    CUDA_CHECK(cudaMemcpy(h_cells.speeds[k], cur->speeds[k],
                          n_cells * sizeof(float), cudaMemcpyDeviceToHost));

  gettimeofday(&timstr, NULL);
  double col_toc = timstr.tv_sec + timstr.tv_usec / 1e6;
  double tot_toc = col_toc;

  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n",
         calc_reynolds(params, &h_cells, h_obstacles));
  printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
  printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
  printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
  printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);

  write_values(params, &h_cells, h_obstacles, av_vels);

  // free memory
  for (int k = 0; k < NSPEEDS; k++) {
    cudaFree(d_cells.speeds[k]);
    cudaFree(d_tmp_cells.speeds[k]);
    free(h_cells.speeds[k]);
  }
  cudaFree(d_obstacles);
  cudaFree(d_vel);
  free(h_obstacles);
  free(av_vels);

  return EXIT_SUCCESS;
}

__global__ void accelerate_flow_kernel(
    float * __restrict__ c1, float * __restrict__ c3,
    float * __restrict__ c5, float * __restrict__ c6,
    float * __restrict__ c7, float * __restrict__ c8,
    const int * __restrict__ obs,
    int nx, int ny, float density, float accel)
{
  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii >= nx) return;

  const int jj  = ny - 2;
  const int idx = jj * nx + ii;

  const float w1 = density * accel / 9.f;
  const float w2 = density * accel / 36.f;

  if (!obs[idx]
      && (c3[idx] - w1) > 0.f
      && (c6[idx] - w2) > 0.f
      && (c7[idx] - w2) > 0.f)
  {
    c1[idx] += w1; c5[idx] += w2; c8[idx] += w2;
    c3[idx] -= w1; c6[idx] -= w2; c7[idx] -= w2;
  }
}

__global__ void propagate_collide_kernel(
    const float * __restrict__ c0, const float * __restrict__ c1,
    const float * __restrict__ c2, const float * __restrict__ c3,
    const float * __restrict__ c4, const float * __restrict__ c5,
    const float * __restrict__ c6, const float * __restrict__ c7,
    const float * __restrict__ c8,
    float * __restrict__ out0, float * __restrict__ out1,
    float * __restrict__ out2, float * __restrict__ out3,
    float * __restrict__ out4, float * __restrict__ out5,
    float * __restrict__ out6, float * __restrict__ out7,
    float * __restrict__ out8,
    const int * __restrict__ obs,
    float * __restrict__ vel_out,
    int nx, int ny, float omega)
{
  const int ii = blockIdx.x * blockDim.x + threadIdx.x;
  const int jj = blockIdx.y * blockDim.y + threadIdx.y;
  if (ii >= nx || jj >= ny) return;

  const int x_w = (ii == 0)      ? nx - 1 : ii - 1;
  const int x_e = (ii == nx - 1) ? 0      : ii + 1;
  const int y_s = (jj == 0)      ? ny - 1 : jj - 1;
  const int y_n = (jj == ny - 1) ? 0      : jj + 1;

  const int idx   = jj  * nx + ii;
  const int row   = jj  * nx;
  const int row_s = y_s * nx;
  const int row_n = y_n * nx;

  const float r0 = c0[idx];
  const float r1 = c1[row   + x_w];
  const float r2 = c2[row_s + ii];
  const float r3 = c3[row   + x_e];
  const float r4 = c4[row_n + ii];
  const float r5 = c5[row_s + x_w];
  const float r6 = c6[row_s + x_e];
  const float r7 = c7[row_n + x_e];
  const float r8 = c8[row_n + x_w];

  const float rho     = fmaxf(r0+r1+r2+r3+r4+r5+r6+r7+r8, 1e-20f);
  const float inv_rho = 1.f / rho;
  const float ux      = (r1+r5+r8 - (r3+r6+r7)) * inv_rho;
  const float uy      = (r2+r5+r6 - (r4+r7+r8)) * inv_rho;
  const float u2      = ux*ux + uy*uy;
  const float com     = 1.f - 1.5f * u2;

  const float w0c = 4.f  / 9.f;
  const float w1c = 1.f  / 9.f;
  const float w2c = 1.f  / 36.f;

  float feq0 = w0c * rho * com;
  float feq1 = w1c * rho * (com + 3.f*ux + 4.5f*ux*ux);
  float feq2 = w1c * rho * (com + 3.f*uy + 4.5f*uy*uy);
  float feq3 = w1c * rho * (com - 3.f*ux + 4.5f*ux*ux);
  float feq4 = w1c * rho * (com - 3.f*uy + 4.5f*uy*uy);
  float uxy;
  uxy =  ux + uy; float feq5 = w2c * rho * (com + 3.f*uxy + 4.5f*uxy*uxy);
  uxy = -ux + uy; float feq6 = w2c * rho * (com + 3.f*uxy + 4.5f*uxy*uxy);
  uxy = -ux - uy; float feq7 = w2c * rho * (com + 3.f*uxy + 4.5f*uxy*uxy);
  uxy =  ux - uy; float feq8 = w2c * rho * (com + 3.f*uxy + 4.5f*uxy*uxy);

  float col0 = r0 + omega * (feq0 - r0);
  float col1 = r1 + omega * (feq1 - r1);
  float col2 = r2 + omega * (feq2 - r2);
  float col3 = r3 + omega * (feq3 - r3);
  float col4 = r4 + omega * (feq4 - r4);
  float col5 = r5 + omega * (feq5 - r5);
  float col6 = r6 + omega * (feq6 - r6);
  float col7 = r7 + omega * (feq7 - r7);
  float col8 = r8 + omega * (feq8 - r8);

  const float ob = (float)obs[idx];

  out0[idx] = col0 + ob * (r0 - col0);
  out1[idx] = col1 + ob * (r3 - col1);
  out2[idx] = col2 + ob * (r4 - col2);
  out3[idx] = col3 + ob * (r1 - col3);
  out4[idx] = col4 + ob * (r2 - col4);
  out5[idx] = col5 + ob * (r7 - col5);
  out6[idx] = col6 + ob * (r8 - col6);
  out7[idx] = col7 + ob * (r5 - col7);
  out8[idx] = col8 + ob * (r6 - col8);

  vel_out[idx] = (1.f - ob) * sqrtf(u2);
}

float calc_reynolds(const t_param params,
                         const t_speed *h_cells, const int *h_obstacles)
{
  const int nx = params.nx;
  const int ny = params.ny;

  float tot_u = 0.f;
  int   tot_c = 0;

  for (int jj = 0; jj < ny; jj++) {
    for (int ii = 0; ii < nx; ii++) {
      const int idx = jj * nx + ii;
      if (h_obstacles[idx]) continue;

      float rho = 0.f;
      for (int k = 0; k < NSPEEDS; k++) rho += h_cells->speeds[k][idx];
      const float inv = 1.f / rho;

      const float ux = (h_cells->speeds[1][idx] + h_cells->speeds[5][idx] + h_cells->speeds[8][idx]
                      -(h_cells->speeds[3][idx] + h_cells->speeds[6][idx] + h_cells->speeds[7][idx])) * inv;
      const float uy = (h_cells->speeds[2][idx] + h_cells->speeds[5][idx] + h_cells->speeds[6][idx]
                      -(h_cells->speeds[4][idx] + h_cells->speeds[7][idx] + h_cells->speeds[8][idx])) * inv;
      tot_u += sqrtf(ux*ux + uy*uy);
      tot_c++;
    }
  }

  const float visc = 1.f / 6.f * (2.f / params.omega - 1.f);
  const float av   = (tot_c > 0) ? tot_u / (float)tot_c : 0.f;
  return av * params.reynolds_dim / visc;
}

int initialise(const char *paramfile, const char *obstaclefile,
               t_param *params,
               t_speed *h_cells, int **h_obstacles_ptr,
               float **av_vels_ptr)
{
  char message[1024];
  FILE *fp = fopen(paramfile, "r");
  if (!fp) {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  int ret;
  ret = fscanf(fp, "%d\n", &params->nx);           if (ret != 1) die("bad nx",          __LINE__, __FILE__);
  ret = fscanf(fp, "%d\n", &params->ny);           if (ret != 1) die("bad ny",          __LINE__, __FILE__);
  ret = fscanf(fp, "%d\n", &params->maxIters);     if (ret != 1) die("bad maxIters",    __LINE__, __FILE__);
  ret = fscanf(fp, "%d\n", &params->reynolds_dim); if (ret != 1) die("bad reynolds_dim",__LINE__, __FILE__);
  ret = fscanf(fp, "%f\n", &params->density);      if (ret != 1) die("bad density",     __LINE__, __FILE__);
  ret = fscanf(fp, "%f\n", &params->accel);        if (ret != 1) die("bad accel",       __LINE__, __FILE__);
  ret = fscanf(fp, "%f\n", &params->omega);        if (ret != 1) die("bad omega",       __LINE__, __FILE__);
  fclose(fp);

  const int n_cells = params->nx * params->ny;

  for (int k = 0; k < NSPEEDS; k++) {
    h_cells->speeds[k] = (float *)calloc(n_cells, sizeof(float));
    if (!h_cells->speeds[k]) die("cannot allocate host speeds", __LINE__, __FILE__);
  }

  *h_obstacles_ptr = (int *)calloc(n_cells, sizeof(int));
  if (!*h_obstacles_ptr) die("cannot allocate host obstacles", __LINE__, __FILE__);

  const float w0 = params->density * 4.f / 9.f;
  const float w1 = params->density      / 9.f;
  const float w2 = params->density      / 36.f;

  for (int i = 0; i < n_cells; i++) {
    h_cells->speeds[0][i] = w0;
    h_cells->speeds[1][i] = w1;
    h_cells->speeds[2][i] = w1;
    h_cells->speeds[3][i] = w1;
    h_cells->speeds[4][i] = w1;
    h_cells->speeds[5][i] = w2;
    h_cells->speeds[6][i] = w2;
    h_cells->speeds[7][i] = w2;
    h_cells->speeds[8][i] = w2;
  }

  fp = fopen(obstaclefile, "r");
  if (!fp) {
    sprintf(message, "could not open obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  int xx, yy, blocked;
  while ((ret = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF) {
    if (ret != 3)                        die("bad obstacle line",       __LINE__, __FILE__);
    if (xx < 0 || xx > params->nx - 1)   die("obstacle x out of range", __LINE__, __FILE__);
    if (yy < 0 || yy > params->ny - 1)   die("obstacle y out of range", __LINE__, __FILE__);
    if (blocked != 1)                    die("blocked must be 1",       __LINE__, __FILE__);
    (*h_obstacles_ptr)[xx + yy * params->nx] = blocked;
  }
  fclose(fp);

  *av_vels_ptr = (float *)malloc(params->maxIters * sizeof(float));
  if (!*av_vels_ptr) die("cannot allocate av_vels", __LINE__, __FILE__);

  return EXIT_SUCCESS;
}

int write_values(const t_param params, const t_speed *h_cells,
                 const int *h_obstacles, const float *av_vels)
{
  const float c_sq = 1.f / 3.f;

  FILE *fp = fopen(FINALSTATEFILE, "w");
  if (!fp) die("could not open final_state.dat", __LINE__, __FILE__);

  for (int jj = 0; jj < params.ny; jj++) {
    for (int ii = 0; ii < params.nx; ii++) {
      const int idx = ii + jj * params.nx;
      float u_x, u_y, u, pressure;

      if (h_obstacles[idx]) {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      } else {
        float ld = 0.f;
        for (int k = 0; k < NSPEEDS; k++) ld += h_cells->speeds[k][idx];
        u_x = (h_cells->speeds[1][idx] + h_cells->speeds[5][idx] + h_cells->speeds[8][idx]
             -(h_cells->speeds[3][idx] + h_cells->speeds[6][idx] + h_cells->speeds[7][idx])) / ld;
        u_y = (h_cells->speeds[2][idx] + h_cells->speeds[5][idx] + h_cells->speeds[6][idx]
             -(h_cells->speeds[4][idx] + h_cells->speeds[7][idx] + h_cells->speeds[8][idx])) / ld;
        u = sqrtf(u_x*u_x + u_y*u_y);
        pressure = ld * c_sq;
      }

      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n",
              ii, jj, u_x, u_y, u, pressure, h_obstacles[idx]);
    }
  }
  fclose(fp);

  fp = fopen(AVVELSFILE, "w");
  if (!fp) die("could not open av_vels.dat", __LINE__, __FILE__);
  for (int tt = 0; tt < params.maxIters; tt++)
    fprintf(fp, "%d:\t%.12E\n", tt, av_vels[tt]);
  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char *message, const int line, const char *file)
{
  fprintf(stderr, "Error at line %d of file %s:\n%s\n", line, file, message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char *exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}
