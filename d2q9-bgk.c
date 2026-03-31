/*
** Optimised serial d2q9-bgk lattice Boltzmann code.
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"
#define ALIGNMENT       64

typedef struct
{
  int    nx;
  int    ny;
  int    maxIters;
  int    reynolds_dim;
  float density;
  float accel;
  float omega;
} t_param;

typedef struct
{
  float *speeds[NSPEEDS];
} t_speed;

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr);

int accelerate_flow(const t_param params, t_speed* cells, int* obstacles);

float propagate_rebound_collide(
    const t_param params,
    const t_speed *cells, t_speed *tmp_cells,
    const int *obstacles);

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels);

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr);

float total_density(const t_param params, t_speed* cells);
float calc_reynolds(const t_param params, t_speed* cells, int* obstacles);

void die(const char* message, const int line, const char* file);
void usage(const char* exe);

static inline float *alloc_floats(int n)
{
  float *p = NULL;
  if (posix_memalign((void **)&p, ALIGNMENT, n * sizeof(float)) != 0)
    return NULL;
  memset(p, 0, n * sizeof(float));
  return p;
}

int main(int argc, char* argv[])
{
  char*    paramfile = NULL;
  char*    obstaclefile = NULL;
  t_param  params;
  t_speed* cells     = NULL;
  t_speed* tmp_cells = NULL;
  int*     obstacles = NULL;
  float*   av_vels   = NULL;
  struct timeval timstr;
  double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc;

  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  gettimeofday(&timstr, NULL);
  tot_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  init_tic = tot_tic;
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels);

  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic = init_toc;

  for (int tt = 0; tt < params.maxIters; tt++)
  {
    accelerate_flow(params, cells, obstacles);
    av_vels[tt] = propagate_rebound_collide(params, cells, tmp_cells, obstacles);

    t_speed temp = *cells;
    *cells = *tmp_cells;
    *tmp_cells = temp;

#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif
  }

  gettimeofday(&timstr, NULL);
  comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  col_tic = comp_toc;

  gettimeofday(&timstr, NULL);
  col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  tot_toc = col_toc;

  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
  printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
  printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
  printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
  printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);
  write_values(params, cells, obstacles, av_vels);
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);

  return EXIT_SUCCESS;
}

int accelerate_flow(const t_param params, t_speed* restrict cells, int* restrict obstacles)
{
  const float w1 = params.density * params.accel / 9.f;
  const float w2 = params.density * params.accel / 36.f;

  const int jj  = params.ny - 2;
  const int nx  = params.nx;
  const int row = jj * nx;

  for (int ii = 0; ii < nx; ii++)
  {
    const int idx = ii + row;

    if (!obstacles[idx]
        && (cells->speeds[3][idx] - w1) > 0.f
        && (cells->speeds[6][idx] - w2) > 0.f
        && (cells->speeds[7][idx] - w2) > 0.f)
    {
      cells->speeds[1][idx] += w1;
      cells->speeds[5][idx] += w2;
      cells->speeds[8][idx] += w2;
      cells->speeds[3][idx] -= w1;
      cells->speeds[6][idx] -= w2;
      cells->speeds[7][idx] -= w2;
    }
  }

  return EXIT_SUCCESS;
}

float propagate_rebound_collide(
    const t_param params,
    const t_speed *cells, t_speed *tmp_cells,
    const int *obstacles)
{
  const int nx = params.nx;
  const int ny = params.ny;
  const float omega = params.omega;

  const float * restrict c0 = cells->speeds[0];
  const float * restrict c1 = cells->speeds[1];
  const float * restrict c2 = cells->speeds[2];
  const float * restrict c3 = cells->speeds[3];
  const float * restrict c4 = cells->speeds[4];
  const float * restrict c5 = cells->speeds[5];
  const float * restrict c6 = cells->speeds[6];
  const float * restrict c7 = cells->speeds[7];
  const float * restrict c8 = cells->speeds[8];

  float * restrict out0 = tmp_cells->speeds[0];
  float * restrict out1 = tmp_cells->speeds[1];
  float * restrict out2 = tmp_cells->speeds[2];
  float * restrict out3 = tmp_cells->speeds[3];
  float * restrict out4 = tmp_cells->speeds[4];
  float * restrict out5 = tmp_cells->speeds[5];
  float * restrict out6 = tmp_cells->speeds[6];
  float * restrict out7 = tmp_cells->speeds[7];
  float * restrict out8 = tmp_cells->speeds[8];

  const int * restrict obs = obstacles;

  const float w0c = 4.f / 9.f;
  const float w1c = 1.f / 9.f;
  const float w2c = 1.f / 36.f;

  float acc_u = 0.f;
  int   tot_cells = 0;

  for (int jj = 0; jj < ny; ++jj)
  {
    const int row   = jj * nx;
    const int row_s = ((jj == 0) ? ny - 1 : jj - 1) * nx;
    const int row_n = ((jj == ny - 1) ? 0 : jj + 1) * nx;

    {
      const int idx = row;
      const float r0=c0[idx],     r1=c1[row+nx-1],  r2=c2[row_s],
                  r3=c3[row+1],   r4=c4[row_n],     r5=c5[row_s+nx-1],
                  r6=c6[row_s+1], r7=c7[row_n+1],   r8=c8[row_n+nx-1];

      const float rho = fmaxf(r0+r1+r2+r3+r4+r5+r6+r7+r8, 1e-20f);
      const float inv_rho = 1.f / rho;
      const float ux = (r1+r5+r8-(r3+r6+r7))*inv_rho;
      const float uy = (r2+r5+r6-(r4+r7+r8))*inv_rho;
      const float u2 = ux*ux+uy*uy;
      const float com = 1.f-1.5f*u2;

      if (obs[idx]) {
        out0[idx]=r0; out1[idx]=r3; out2[idx]=r4; out3[idx]=r1; out4[idx]=r2;
        out5[idx]=r7; out6[idx]=r8; out7[idx]=r5; out8[idx]=r6;
      } else {
        float uxy;
        out0[idx]=r0+omega*(w0c*rho*com-r0);
        out1[idx]=r1+omega*(w1c*rho*(com+3.f*ux+4.5f*ux*ux)-r1);
        out2[idx]=r2+omega*(w1c*rho*(com+3.f*uy+4.5f*uy*uy)-r2);
        out3[idx]=r3+omega*(w1c*rho*(com-3.f*ux+4.5f*ux*ux)-r3);
        out4[idx]=r4+omega*(w1c*rho*(com-3.f*uy+4.5f*uy*uy)-r4);
        uxy= ux+uy; out5[idx]=r5+omega*(w2c*rho*(com+3.f*uxy+4.5f*uxy*uxy)-r5);
        uxy=-ux+uy; out6[idx]=r6+omega*(w2c*rho*(com+3.f*uxy+4.5f*uxy*uxy)-r6);
        uxy=-ux-uy; out7[idx]=r7+omega*(w2c*rho*(com+3.f*uxy+4.5f*uxy*uxy)-r7);
        uxy= ux-uy; out8[idx]=r8+omega*(w2c*rho*(com+3.f*uxy+4.5f*uxy*uxy)-r8);
        acc_u += sqrtf(u2);
        tot_cells++;
      }
    }

    if (nx > 1)
    {
      const int ii=nx-1, idx=row+ii;
      const float r0=c0[idx],     r1=c1[row+ii-1],  r2=c2[row_s+ii],
                  r3=c3[row],     r4=c4[row_n+ii],  r5=c5[row_s+ii-1],
                  r6=c6[row_s],   r7=c7[row_n],     r8=c8[row_n+ii-1];

      const float rho = fmaxf(r0+r1+r2+r3+r4+r5+r6+r7+r8, 1e-20f);
      const float inv_rho = 1.f / rho;
      const float ux = (r1+r5+r8-(r3+r6+r7))*inv_rho;
      const float uy = (r2+r5+r6-(r4+r7+r8))*inv_rho;
      const float u2 = ux*ux+uy*uy;
      const float com = 1.f-1.5f*u2;

      if (obs[idx]) {
        out0[idx]=r0; out1[idx]=r3; out2[idx]=r4; out3[idx]=r1; out4[idx]=r2;
        out5[idx]=r7; out6[idx]=r8; out7[idx]=r5; out8[idx]=r6;
      } else {
        float uxy;
        out0[idx]=r0+omega*(w0c*rho*com-r0);
        out1[idx]=r1+omega*(w1c*rho*(com+3.f*ux+4.5f*ux*ux)-r1);
        out2[idx]=r2+omega*(w1c*rho*(com+3.f*uy+4.5f*uy*uy)-r2);
        out3[idx]=r3+omega*(w1c*rho*(com-3.f*ux+4.5f*ux*ux)-r3);
        out4[idx]=r4+omega*(w1c*rho*(com-3.f*uy+4.5f*uy*uy)-r4);
        uxy= ux+uy; out5[idx]=r5+omega*(w2c*rho*(com+3.f*uxy+4.5f*uxy*uxy)-r5);
        uxy=-ux+uy; out6[idx]=r6+omega*(w2c*rho*(com+3.f*uxy+4.5f*uxy*uxy)-r6);
        uxy=-ux-uy; out7[idx]=r7+omega*(w2c*rho*(com+3.f*uxy+4.5f*uxy*uxy)-r7);
        uxy= ux-uy; out8[idx]=r8+omega*(w2c*rho*(com+3.f*uxy+4.5f*uxy*uxy)-r8);
        acc_u += sqrtf(u2);
        tot_cells++;
      }
    }

    #pragma omp simd reduction(+:acc_u) reduction(+:tot_cells)
    for (int ii = 1; ii < nx - 1; ++ii)
    {
      const int idx = row + ii;

      const float r0 = c0[idx];
      const float r1 = c1[row   + ii - 1];
      const float r2 = c2[row_s + ii];
      const float r3 = c3[row   + ii + 1];
      const float r4 = c4[row_n + ii];
      const float r5 = c5[row_s + ii - 1];
      const float r6 = c6[row_s + ii + 1];
      const float r7 = c7[row_n + ii + 1];
      const float r8 = c8[row_n + ii - 1];

      const float rho = fmaxf(r0+r1+r2+r3+r4+r5+r6+r7+r8, 1e-20f);
      const float inv_rho = 1.f / rho;
      const float ux = (r1+r5+r8 - (r3+r6+r7)) * inv_rho;
      const float uy = (r2+r5+r6 - (r4+r7+r8)) * inv_rho;
      const float u2 = ux*ux + uy*uy;
      const float com = 1.f - 1.5f * u2;

      float feq0 = w0c*rho*com;
      float feq1 = w1c*rho*(com + 3.f*ux  + 4.5f*ux*ux);
      float feq2 = w1c*rho*(com + 3.f*uy  + 4.5f*uy*uy);
      float feq3 = w1c*rho*(com - 3.f*ux  + 4.5f*ux*ux);
      float feq4 = w1c*rho*(com - 3.f*uy  + 4.5f*uy*uy);
      float uxy;
      uxy= ux+uy; float feq5 = w2c*rho*(com+3.f*uxy+4.5f*uxy*uxy);
      uxy=-ux+uy; float feq6 = w2c*rho*(com+3.f*uxy+4.5f*uxy*uxy);
      uxy=-ux-uy; float feq7 = w2c*rho*(com+3.f*uxy+4.5f*uxy*uxy);
      uxy= ux-uy; float feq8 = w2c*rho*(com+3.f*uxy+4.5f*uxy*uxy);

      float col0=r0+omega*(feq0-r0), col1=r1+omega*(feq1-r1);
      float col2=r2+omega*(feq2-r2), col3=r3+omega*(feq3-r3);
      float col4=r4+omega*(feq4-r4), col5=r5+omega*(feq5-r5);
      float col6=r6+omega*(feq6-r6), col7=r7+omega*(feq7-r7);
      float col8=r8+omega*(feq8-r8);

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

      acc_u += (1.f - ob) * sqrtf(u2);
      tot_cells += (1 - obs[idx]);
    }
  }

  return (tot_cells > 0) ? acc_u / (float)tot_cells : 0.f;
}

float calc_reynolds(const t_param params, t_speed* cells, int* obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);
  return propagate_rebound_collide(params, cells, cells, obstacles)
         * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed* cells)
{
  float total = 0.f;
  const int n = params.nx * params.ny;

  for (int idx = 0; idx < n; idx++)
    for (int kk = 0; kk < NSPEEDS; kk++)
      total += cells->speeds[kk][idx];

  return total;
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr)
{
  char   message[1024];
  FILE*  fp;
  int    xx, yy;
  int    blocked;
  int    retval;

  fp = fopen(paramfile, "r");
  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  retval = fscanf(fp, "%d\n", &(params->nx));
  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);
  retval = fscanf(fp, "%d\n", &(params->ny));
  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);
  retval = fscanf(fp, "%d\n", &(params->maxIters));
  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);
  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));
  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);
  retval = fscanf(fp, "%f\n", &(params->density));
  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);
  retval = fscanf(fp, "%f\n", &(params->accel));
  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);
  retval = fscanf(fp, "%f\n", &(params->omega));
  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  fclose(fp);

  int n_cells = params->nx * params->ny;

  t_speed *cells = malloc(sizeof(t_speed));
  *cells_ptr = cells;
  if (!cells) die("cannot allocate memory for cells", __LINE__, __FILE__);

  t_speed *tmp_cells = malloc(sizeof(t_speed));
  *tmp_cells_ptr = tmp_cells;
  if (!tmp_cells) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  *obstacles_ptr = malloc(sizeof(int) * n_cells);
  if (!*obstacles_ptr) die("cannot allocate memory for obstacles", __LINE__, __FILE__);

  for (int k = 0; k < NSPEEDS; k++) {
    cells->speeds[k]     = alloc_floats(n_cells);
    tmp_cells->speeds[k] = alloc_floats(n_cells);
    if (!cells->speeds[k] || !tmp_cells->speeds[k])
      die("cannot allocate speeds", __LINE__, __FILE__);
  }

  const float w0 = params->density * 4.f / 9.f;
  const float w1 = params->density      / 9.f;
  const float w2 = params->density      / 36.f;

  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      int idx = ii + jj * params->nx;
      cells->speeds[0][idx] = w0;
      cells->speeds[1][idx] = w1;
      cells->speeds[2][idx] = w1;
      cells->speeds[3][idx] = w1;
      cells->speeds[4][idx] = w1;
      cells->speeds[5][idx] = w2;
      cells->speeds[6][idx] = w2;
      cells->speeds[7][idx] = w2;
      cells->speeds[8][idx] = w2;
    }
  }

  for (int jj = 0; jj < params->ny; jj++)
    for (int ii = 0; ii < params->nx; ii++)
      (*obstacles_ptr)[ii + jj*params->nx] = 0;

  fp = fopen(obstaclefile, "r");
  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);
    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);
    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);
    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;
  }

  fclose(fp);

  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr)
{
  for (int k = 0; k < NSPEEDS; k++) {
    free((*cells_ptr)->speeds[k]);
    free((*tmp_cells_ptr)->speeds[k]);
  }

  free(*cells_ptr);     *cells_ptr     = NULL;
  free(*tmp_cells_ptr); *tmp_cells_ptr = NULL;
  free(*obstacles_ptr); *obstacles_ptr = NULL;
  free(*av_vels_ptr);   *av_vels_ptr   = NULL;

  return EXIT_SUCCESS;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels)
{
  FILE* fp;
  const float c_sq = 1.f / 3.f;
  float local_density;
  float pressure;
  float u_x, u_y, u;

  fp = fopen(FINALSTATEFILE, "w");
  if (fp == NULL)
    die("could not open file output file", __LINE__, __FILE__);

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      const int idx = ii + jj * params.nx;

      if (obstacles[idx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      else
      {
        local_density = 0.f;
        for (int kk = 0; kk < NSPEEDS; kk++)
          local_density += cells->speeds[kk][idx];

        u_x = (cells->speeds[1][idx] + cells->speeds[5][idx] + cells->speeds[8][idx]
             - (cells->speeds[3][idx] + cells->speeds[6][idx] + cells->speeds[7][idx]))
              / local_density;

        u_y = (cells->speeds[2][idx] + cells->speeds[5][idx] + cells->speeds[6][idx]
             - (cells->speeds[4][idx] + cells->speeds[7][idx] + cells->speeds[8][idx]))
              / local_density;

        u = sqrtf(u_x * u_x + u_y * u_y);
        pressure = local_density * c_sq;
      }

      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n",
              ii, jj, u_x, u_y, u, pressure, obstacles[idx]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");
  if (fp == NULL)
    die("could not open file output file", __LINE__, __FILE__);

  for (int ii = 0; ii < params.maxIters; ii++)
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}