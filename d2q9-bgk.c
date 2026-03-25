/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
  float *speeds[NSPEEDS];
} t_speed;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr);

int timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int accelerate_flow(const t_param params, t_speed* cells, int* obstacles);

float propagate_rebound_collide(
    const t_param params,
    const float * restrict c0,
    const float * restrict c1,
    const float * restrict c2,
    const float * restrict c3,
    const float * restrict c4,
    const float * restrict c5,
    const float * restrict c6,
    const float * restrict c7,
    const float * restrict c8,
    float * restrict out0,
    float * restrict out1,
    float * restrict out2,
    float * restrict out3,
    float * restrict out4,
    float * restrict out5,
    float * restrict out6,
    float * restrict out7,
    float * restrict out8,
    const int * restrict obstacles
);

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* cells);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed* cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_speed* cells     = NULL;    /* grid containing fluid densities */
  t_speed* tmp_cells = NULL;    /* scratch space */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;
  double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc;

  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* Total/init time starts here */
  gettimeofday(&timstr, NULL);
  tot_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  init_tic = tot_tic;
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels);

  /* Init time stops here, compute time starts */
  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic = init_toc;

  for (int tt = 0; tt < params.maxIters; tt++)
  {
    accelerate_flow(params, cells, obstacles);

    /* single fused pass: propagate + rebound + collide + av_velocity */
    av_vels[tt] = propagate_rebound_collide(params,
        cells->speeds[0], cells->speeds[1], cells->speeds[2],
        cells->speeds[3], cells->speeds[4], cells->speeds[5],
        cells->speeds[6], cells->speeds[7], cells->speeds[8],
        tmp_cells->speeds[0], tmp_cells->speeds[1], tmp_cells->speeds[2],
        tmp_cells->speeds[3], tmp_cells->speeds[4], tmp_cells->speeds[5],
        tmp_cells->speeds[6], tmp_cells->speeds[7], tmp_cells->speeds[8],
        obstacles);

    /* swap pointers */
    t_speed temp = *cells;
    *cells = *tmp_cells;
    *tmp_cells = temp;

#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif
  }

  /* Compute time stops here, collate time starts */
  gettimeofday(&timstr, NULL);
  comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  col_tic = comp_toc;

  /* Total/collate time stops here */
  gettimeofday(&timstr, NULL);
  col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  tot_toc = col_toc;

  /* write final values and free memory */
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
  const int row = jj * params.nx;

  for (int ii = 0; ii < params.nx; ii++)
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


/*
** Perform BGK collision on a single non-obstacle cell.
** Writes post-collision distributions into o0..o8.
*/
static inline void collide(
    float omega,
    float s0, float s1, float s2, float s3, float s4,
    float s5, float s6, float s7, float s8,
    float *o0, float *o1, float *o2, float *o3, float *o4,
    float *o5, float *o6, float *o7, float *o8)
{
  const float w0 = 4.f / 9.f;
  const float w1 = 1.f / 9.f;
  const float w2 = 1.f / 36.f;

  float rho = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8;
  if (rho <= 1e-20f) rho = 1e-20f;
  const float inv_rho = 1.f / rho;

  const float ux = (s1 + s5 + s8 - (s3 + s6 + s7)) * inv_rho;
  const float uy = (s2 + s5 + s6 - (s4 + s7 + s8)) * inv_rho;

  const float u2     = ux * ux + uy * uy;
  const float common = 1.f - 1.5f * u2;

  float feq0 = w0 * rho * common;
  float feq1 = w1 * rho * (common + 3.f * ux + 4.5f * ux * ux);
  float feq2 = w1 * rho * (common + 3.f * uy + 4.5f * uy * uy);
  float feq3 = w1 * rho * (common - 3.f * ux + 4.5f * ux * ux);
  float feq4 = w1 * rho * (common - 3.f * uy + 4.5f * uy * uy);

  float uxy;
  uxy = ux + uy;
  float feq5 = w2 * rho * (common + 3.f * uxy + 4.5f * uxy * uxy);
  uxy = -ux + uy;
  float feq6 = w2 * rho * (common + 3.f * uxy + 4.5f * uxy * uxy);
  uxy = -ux - uy;
  float feq7 = w2 * rho * (common + 3.f * uxy + 4.5f * uxy * uxy);
  uxy = ux - uy;
  float feq8 = w2 * rho * (common + 3.f * uxy + 4.5f * uxy * uxy);

  *o0 = s0 + omega * (feq0 - s0);
  *o1 = s1 + omega * (feq1 - s1);
  *o2 = s2 + omega * (feq2 - s2);
  *o3 = s3 + omega * (feq3 - s3);
  *o4 = s4 + omega * (feq4 - s4);
  *o5 = s5 + omega * (feq5 - s5);
  *o6 = s6 + omega * (feq6 - s6);
  *o7 = s7 + omega * (feq7 - s7);
  *o8 = s8 + omega * (feq8 - s8);
}


/*
** Single fused pass over the entire grid:
**   1. Stream (propagate) from neighbours using periodic boundary wrapping
**   2. Bounce-back for obstacle cells
**   3. BGK collision for fluid cells
**   4. Accumulate average velocity
**
** Returns the average velocity for this timestep.
*/
float propagate_rebound_collide(
    const t_param params,
    const float * restrict c0,
    const float * restrict c1,
    const float * restrict c2,
    const float * restrict c3,
    const float * restrict c4,
    const float * restrict c5,
    const float * restrict c6,
    const float * restrict c7,
    const float * restrict c8,
    float * restrict out0,
    float * restrict out1,
    float * restrict out2,
    float * restrict out3,
    float * restrict out4,
    float * restrict out5,
    float * restrict out6,
    float * restrict out7,
    float * restrict out8,
    const int * restrict obstacles)
{
  const int   nx    = params.nx;
  const int   ny    = params.ny;
  const float omega = params.omega;

  float tot_u    = 0.f;
  int   tot_cells = 0;

  for (int jj = 0; jj < ny; ++jj)
  {
    const int row   = jj * nx;
    const int row_s = ((jj == 0) ? ny - 1 : jj - 1) * nx;
    const int row_n = ((jj == ny - 1) ? 0 : jj + 1) * nx;

    for (int ii = 0; ii < nx; ++ii)
    {
      const int ii_w = (ii == 0) ? nx - 1 : ii - 1;
      const int ii_e = (ii == nx - 1) ? 0 : ii + 1;
      const int idx  = row + ii;

      /* --- 1. Stream: gather from neighbouring cells --- */
      const float r0 = c0[idx];
      const float r1 = c1[row   + ii_w];
      const float r2 = c2[row_s + ii];
      const float r3 = c3[row   + ii_e];
      const float r4 = c4[row_n + ii];
      const float r5 = c5[row_s + ii_w];
      const float r6 = c6[row_s + ii_e];
      const float r7 = c7[row_n + ii_e];
      const float r8 = c8[row_n + ii_w];

      /* --- 2. Bounce-back for obstacles --- */
      const int obs = obstacles[idx];

      if (obs)
      {
        /* Reverse directions */
        out0[idx] = r0;
        out1[idx] = r3;
        out2[idx] = r4;
        out3[idx] = r1;
        out4[idx] = r2;
        out5[idx] = r7;
        out6[idx] = r8;
        out7[idx] = r5;
        out8[idx] = r6;
      }
      else
      {
        /* --- 3. BGK collision --- */
        collide(omega,
                r0, r1, r2, r3, r4, r5, r6, r7, r8,
                &out0[idx], &out1[idx], &out2[idx], &out3[idx], &out4[idx],
                &out5[idx], &out6[idx], &out7[idx], &out8[idx]);

        /* --- 4. Accumulate average velocity --- */
        const float rho = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8;
        const float inv_rho = 1.f / rho;
        const float ux = (r1 + r5 + r8 - (r3 + r6 + r7)) * inv_rho;
        const float uy = (r2 + r5 + r6 - (r4 + r7 + r8)) * inv_rho;
        tot_u += sqrtf(ux * ux + uy * uy);
        tot_cells++;
      }
    }
  }

  return (tot_cells > 0) ? tot_u / (float)tot_cells : 0.f;
}


float av_velocity(const t_param params,
              float * restrict c0,
              float * restrict c1,
              float * restrict c2,
              float * restrict c3,
              float * restrict c4,
              float * restrict c5,
              float * restrict c6,
              float * restrict c7,
              float * restrict c8,
              const int* obstacles)
{
  const int n = params.nx * params.ny;

  float tot_u    = 0.f;
  int   tot_cells = 0;

  for (int idx = 0; idx < n; ++idx)
  {
    if (obstacles[idx]) continue;

    const float s0 = c0[idx], s1 = c1[idx], s2 = c2[idx];
    const float s3 = c3[idx], s4 = c4[idx], s5 = c5[idx];
    const float s6 = c6[idx], s7 = c7[idx], s8 = c8[idx];

    const float rho     = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8;
    const float inv_rho = 1.f / rho;

    const float ux = (s1 + s5 + s8 - (s3 + s6 + s7)) * inv_rho;
    const float uy = (s2 + s5 + s6 - (s4 + s7 + s8)) * inv_rho;

    tot_u += sqrtf(ux * ux + uy * uy);
    tot_cells++;
  }

  return (tot_cells > 0) ? tot_u / (float)tot_cells : 0.f;
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
  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  t_speed *tmp_cells = malloc(sizeof(t_speed));
  *tmp_cells_ptr = tmp_cells;
  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  *obstacles_ptr = malloc(sizeof(int) * n_cells);
  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  for (int k = 0; k < NSPEEDS; k++)
  {
    cells->speeds[k]     = malloc(n_cells * sizeof(float));
    tmp_cells->speeds[k] = malloc(n_cells * sizeof(float));
    if (!cells->speeds[k] || !tmp_cells->speeds[k])
      die("cannot allocate speeds", __LINE__, __FILE__);
  }

  /* initialise densities */
  const float w0 = params->density * 4.f / 9.f;
  const float w1 = params->density       / 9.f;
  const float w2 = params->density       / 36.f;

  for (int idx = 0; idx < n_cells; idx++)
  {
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

  /* set all cells in obstacle array to zero */
  for (int idx = 0; idx < n_cells; idx++)
    (*obstacles_ptr)[idx] = 0;

  /* open the obstacle data file */
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
    (*obstacles_ptr)[xx + yy * params->nx] = blocked;
  }

  fclose(fp);

  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr)
{
  for (int k = 0; k < NSPEEDS; k++)
  {
    free((*cells_ptr)->speeds[k]);
    free((*tmp_cells_ptr)->speeds[k]);
  }

  free(*cells_ptr);
  *cells_ptr = NULL;

  free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, t_speed* cells, int* obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params,
          cells->speeds[0], cells->speeds[1], cells->speeds[2],
          cells->speeds[3], cells->speeds[4], cells->speeds[5],
          cells->speeds[6], cells->speeds[7], cells->speeds[8],
          obstacles) * params.reynolds_dim / viscosity;
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