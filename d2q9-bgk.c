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

#define IDX(ii,jj,nx) ((ii) + (jj)*(nx))

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
  float* speeds[NSPEEDS];
} t_speed;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed* cells, t_speed* tmp_cells,
               int** obstacles_ptr, float** av_vels_ptr);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
int timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int accelerate_flow(
    const t_param params,
    float * restrict c1,
    float * restrict c3,
    float * restrict c5,
    float * restrict c6,
    float * restrict c7,
    float * restrict c8,
    const int * restrict obstacles
);
int fused_kernal(
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

float total_density(const t_param params, t_speed* cells);
float av_velocity(const t_param params, t_speed* cells, int* obstacles);
float calc_reynolds(const t_param params, t_speed* cells, int* obstacles);

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels);
/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed* cells, t_speed* tmp_cells,
             int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed* cells, int* obstacles);

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
  t_speed cells;    /* grid containing fluid densities */
  t_speed tmp_cells;    /* scratch space */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;                                                             /* structure to hold elapsed time */
  double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc; /* floating point numbers to calculate elapsed wallclock time */

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

  /* Total/init time starts here: initialise our data structures and load values from file */
  gettimeofday(&timstr, NULL);
  tot_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  init_tic=tot_tic;
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels);

  /* Init time stops here, compute time starts*/
  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic=init_toc;

  for (int tt = 0; tt < params.maxIters; tt++)
  {
    timestep(params, &cells, &tmp_cells, obstacles);
    av_vels[tt] = av_velocity(params, &cells, obstacles);
#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif
  }
  
  /* Compute time stops here, collate time starts*/
  gettimeofday(&timstr, NULL);
  comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  col_tic=comp_toc;

  // Collate data from ranks here 

  /* Total/collate time stops here.*/
  gettimeofday(&timstr, NULL);
  col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  tot_toc = col_toc;
  
  /* write final values and free memory */
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, &cells, obstacles));
  printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
  printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
  printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
  printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);
  write_values(params, &cells, obstacles, av_vels);
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);

  return EXIT_SUCCESS;
}

int timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles)
{
  /* Pull out planes so the compiler sees simple restrict pointers */
  float * restrict c0 = cells->speeds[0];
  float * restrict c1 = cells->speeds[1];
  float * restrict c2 = cells->speeds[2];
  float * restrict c3 = cells->speeds[3];
  float * restrict c4 = cells->speeds[4];
  float * restrict c5 = cells->speeds[5];
  float * restrict c6 = cells->speeds[6];
  float * restrict c7 = cells->speeds[7];
  float * restrict c8 = cells->speeds[8];

  float * restrict out0 = tmp_cells->speeds[0];
  float * restrict out1 = tmp_cells->speeds[1];
  float * restrict out2 = tmp_cells->speeds[2];
  float * restrict out3 = tmp_cells->speeds[3];
  float * restrict out4 = tmp_cells->speeds[4];
  float * restrict out5 = tmp_cells->speeds[5];
  float * restrict out6 = tmp_cells->speeds[6];
  float * restrict out7 = tmp_cells->speeds[7];
  float * restrict out8 = tmp_cells->speeds[8];

  const int * restrict obst = obstacles;

  /* accelerate modifies ONLY the current lattice (cells) */
  accelerate_flow(params, c1, c3, c5, c6, c7, c8, obst);

  /* fused kernel reads from c* and writes to out* */
  fused_kernal(params,
               c0, c1, c2, c3, c4, c5, c6, c7, c8,
               out0, out1, out2, out3, out4, out5, out6, out7, out8,
               obst);

  /* swap pointer planes (SoA swap) so new state becomes cells */
  t_speed temp = *cells;
  *cells = *tmp_cells;
  *tmp_cells = temp;

  return EXIT_SUCCESS;
}

int accelerate_flow(
    const t_param params,
    float * restrict c1,
    float * restrict c3,
    float * restrict c5,
    float * restrict c6,
    float * restrict c7,
    float * restrict c8,
    const int * restrict obstacles
){
  const float w1 = params.density * params.accel / 9.f;
  const float w2 = params.density * params.accel / 36.f;
  const int jj = params.ny - 2;
  const int base = jj * params.nx;

  /* No wrap in x here: idx = base + ii */
  #pragma omp simd
  for (int ii = 0; ii < params.nx; ii++)
  {
    const int idx = base + ii;

    /* mask: 1 if apply accel else 0 */
    const float not_obst = 1.0f - (float)(obstacles[idx] != 0);

    const float ok3 = (float)((c3[idx] - w1) > 0.f);
    const float ok6 = (float)((c6[idx] - w2) > 0.f);
    const float ok7 = (float)((c7[idx] - w2) > 0.f);

    const float do_accel = not_obst * ok3 * ok6 * ok7;

    c1[idx] += do_accel * w1;
    c5[idx] += do_accel * w2;
    c8[idx] += do_accel * w2;

    c3[idx] -= do_accel * w1;
    c6[idx] -= do_accel * w2;
    c7[idx] -= do_accel * w2;
  }

  return EXIT_SUCCESS;
}

int fused_kernal(
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
){
  const float c_sq = 1.f / 3.f;
  const float w0   = 4.f / 9.f;
  const float w1   = 1.f / 9.f;
  const float w2   = 1.f / 36.f;

  const float inv_csq      = 1.f / c_sq;
  const float inv_csq2     = inv_csq * inv_csq;
  const float half_inv_csq = 0.5f * inv_csq;

  for (int jj = 0; jj < params.ny; jj++)
  {
    const int y_n = (jj + 1 == params.ny) ? 0 : (jj + 1);
    const int y_s = (jj == 0) ? (params.ny - 1) : (jj - 1);

    const int row   = jj  * params.nx;
    const int row_n = y_n * params.nx;
    const int row_s = y_s * params.nx;

    /* hint SIMD: compilers may or may not accept due to wrap/branch */
    #pragma omp simd
    for (int ii = 0; ii < params.nx; ii++)
    {
      const int x_e = (ii + 1 == params.nx) ? 0 : (ii + 1);
      const int x_w = (ii == 0) ? (params.nx - 1) : (ii - 1);

      const int idx    = row   + ii;
      const int idx_w  = row   + x_w;
      const int idx_e  = row   + x_e;
      const int idx_s  = row_s + ii;
      const int idx_n  = row_n + ii;

      const int idx_ws = row_s + x_w;
      const int idx_es = row_s + x_e;
      const int idx_en = row_n + x_e;
      const int idx_wn = row_n + x_w;

      /* propagate (pull) into locals from old lattice */
      const float t0 = c0[idx];
      const float t1 = c1[idx_w];
      const float t2 = c2[idx_s];
      const float t3 = c3[idx_e];
      const float t4 = c4[idx_n];
      const float t5 = c5[idx_ws];
      const float t6 = c6[idx_es];
      const float t7 = c7[idx_en];
      const float t8 = c8[idx_wn];

      if (obstacles[idx])
      {
        /* rebound (write new lattice to out*) */
        out0[idx] = t0;
        out1[idx] = t3;
        out2[idx] = t4;
        out3[idx] = t1;
        out4[idx] = t2;
        out5[idx] = t7;
        out6[idx] = t8;
        out7[idx] = t5;
        out8[idx] = t6;
      }
      else
      {
        const float local_density = t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8;

        const float u_x = (t1 + t5 + t8 - (t3 + t6 + t7)) / local_density;
        const float u_y = (t2 + t5 + t6 - (t4 + t7 + t8)) / local_density;

        const float u_sq = u_x*u_x + u_y*u_y;

        /* directional u (only 1..8 are used) */
        const float u1 =  u_x;
        const float u2 =  u_y;
        const float u3 = -u_x;
        const float u4 = -u_y;
        const float u5 =  u_x + u_y;
        const float u6 = -u_x + u_y;
        const float u7 = -u_x - u_y;
        const float u8 =  u_x - u_y;

        /* equilibrium */
        const float d0 = w0 * local_density * (1.f - u_sq * half_inv_csq);

        const float d1 = w1 * local_density * (1.f + u1*inv_csq + 0.5f*(u1*u1)*inv_csq2 - u_sq*half_inv_csq);
        const float d2 = w1 * local_density * (1.f + u2*inv_csq + 0.5f*(u2*u2)*inv_csq2 - u_sq*half_inv_csq);
        const float d3 = w1 * local_density * (1.f + u3*inv_csq + 0.5f*(u3*u3)*inv_csq2 - u_sq*half_inv_csq);
        const float d4 = w1 * local_density * (1.f + u4*inv_csq + 0.5f*(u4*u4)*inv_csq2 - u_sq*half_inv_csq);

        const float d5 = w2 * local_density * (1.f + u5*inv_csq + 0.5f*(u5*u5)*inv_csq2 - u_sq*half_inv_csq);
        const float d6 = w2 * local_density * (1.f + u6*inv_csq + 0.5f*(u6*u6)*inv_csq2 - u_sq*half_inv_csq);
        const float d7 = w2 * local_density * (1.f + u7*inv_csq + 0.5f*(u7*u7)*inv_csq2 - u_sq*half_inv_csq);
        const float d8 = w2 * local_density * (1.f + u8*inv_csq + 0.5f*(u8*u8)*inv_csq2 - u_sq*half_inv_csq);

        const float om = params.omega;

        out0[idx] = t0 + om * (d0 - t0);
        out1[idx] = t1 + om * (d1 - t1);
        out2[idx] = t2 + om * (d2 - t2);
        out3[idx] = t3 + om * (d3 - t3);
        out4[idx] = t4 + om * (d4 - t4);
        out5[idx] = t5 + om * (d5 - t5);
        out6[idx] = t6 + om * (d6 - t6);
        out7[idx] = t7 + om * (d7 - t7);
        out8[idx] = t8 + om * (d8 - t8);
      }
    }
  }

  return EXIT_SUCCESS;
}




float av_velocity(const t_param params, t_speed* cells, int* obstacles)
{
  int   tot_cells = 0;
  float tot_u = 0.f;

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      int idx = IDX(ii, jj, params.nx);

      if (!obstacles[idx])
      {
        float local_density = 0.f;
        for (int kk = 0; kk < NSPEEDS; kk++)
          local_density += cells->speeds[kk][idx];

        float u_x = (cells->speeds[1][idx] + cells->speeds[5][idx] + cells->speeds[8][idx]
                    - (cells->speeds[3][idx] + cells->speeds[6][idx] + cells->speeds[7][idx]))
                    / local_density;

        float u_y = (cells->speeds[2][idx] + cells->speeds[5][idx] + cells->speeds[6][idx]
                    - (cells->speeds[4][idx] + cells->speeds[7][idx] + cells->speeds[8][idx]))
                    / local_density;

        tot_u += sqrtf(u_x*u_x + u_y*u_y);
        ++tot_cells;
      }
    }
  }

  return tot_u / (float)tot_cells;
}


int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed* cells, t_speed* tmp_cells,
               int** obstacles_ptr, float** av_vels_ptr)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
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

  /* and close up the file */
  fclose(fp);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */
  int ncells = params->nx * params->ny;

  /* allocate 9 arrays for cells + 9 arrays for tmp_cells */
  for (int k = 0; k < NSPEEDS; k++) {
    cells->speeds[k] = (float*)malloc(sizeof(float) * ncells);
    if (!cells->speeds[k]) die("cannot allocate memory for cells speeds", __LINE__, __FILE__);

    tmp_cells->speeds[k] = (float*)malloc(sizeof(float) * ncells);
    if (!tmp_cells->speeds[k]) die("cannot allocate memory for tmp_cells speeds", __LINE__, __FILE__);
  }

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      int idx = IDX(ii, jj, params->nx);

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

  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed* cells_ptr, t_speed* tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr)
{
  /*
  ** free up allocated memory
  */
  for (int k = 0; k < NSPEEDS; k++) {
    free(cells_ptr->speeds[k]);
    cells_ptr->speeds[k] = NULL;

    free(tmp_cells_ptr->speeds[k]);
    tmp_cells_ptr->speeds[k] = NULL;
  }

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, t_speed* cells, int* obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);
  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}


float total_density(const t_param params, t_speed* cells)
{
  float total = 0.f;
  for (int jj = 0; jj < params.ny; jj++)
    for (int ii = 0; ii < params.nx; ii++)
    {
      int idx = IDX(ii, jj, params.nx);
      for (int kk = 0; kk < NSPEEDS; kk++)
        total += cells->speeds[kk][idx];
    }
  return total;
}


int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* an occupied cell */
      if (obstacles[ii + jj*params.nx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        int idx = IDX(ii, jj, params.nx);

        local_density = 0.f;
        for (int kk = 0; kk < NSPEEDS; kk++)
          local_density += cells->speeds[kk][idx];

        u_x = (cells->speeds[1][idx] + cells->speeds[5][idx] + cells->speeds[8][idx]
              - (cells->speeds[3][idx] + cells->speeds[6][idx] + cells->speeds[7][idx]))
              / local_density;

        u_y = (cells->speeds[2][idx] + cells->speeds[5][idx] + cells->speeds[6][idx]
              - (cells->speeds[4][idx] + cells->speeds[7][idx] + cells->speeds[8][idx]))
              / local_density;

        u = sqrtf(u_x*u_x + u_y*u_y);
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii + params.nx * jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

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
