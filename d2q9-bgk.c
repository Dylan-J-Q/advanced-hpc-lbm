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
int accelerate_flow(const t_param params, t_speed* cells, int* obstacles);
int fused_kernal(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);

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
  accelerate_flow(params, cells, obstacles);
  fused_kernal(params, cells, tmp_cells, obstacles);
  return EXIT_SUCCESS;
}

int accelerate_flow(const t_param params, t_speed* cells, int* obstacles)
{
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

  int jj = params.ny - 2;

  for (int ii = 0; ii < params.nx; ii++)
  {
    int idx = IDX(ii, jj, params.nx);

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


int fused_kernal(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles)
{
  /* ---- LOOP 1: PROPAGATE (cells -> tmp_cells) ---- */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      int y_n = (jj + 1) % params.ny;
      int x_e = (ii + 1) % params.nx;
      int y_s = (jj == 0) ? (params.ny - 1) : (jj - 1);
      int x_w = (ii == 0) ? (params.nx - 1) : (ii - 1);

      int idx    = IDX(ii,  jj,  params.nx);
      int idx_w  = IDX(x_w, jj,  params.nx);
      int idx_e  = IDX(x_e, jj,  params.nx);
      int idx_s  = IDX(ii,  y_s, params.nx);
      int idx_n  = IDX(ii,  y_n, params.nx);

      int idx_ws = IDX(x_w, y_s, params.nx);
      int idx_es = IDX(x_e, y_s, params.nx);
      int idx_en = IDX(x_e, y_n, params.nx);
      int idx_wn = IDX(x_w, y_n, params.nx);

      tmp_cells->speeds[0][idx] = cells->speeds[0][idx];
      tmp_cells->speeds[1][idx] = cells->speeds[1][idx_w];
      tmp_cells->speeds[2][idx] = cells->speeds[2][idx_s];
      tmp_cells->speeds[3][idx] = cells->speeds[3][idx_e];
      tmp_cells->speeds[4][idx] = cells->speeds[4][idx_n];
      tmp_cells->speeds[5][idx] = cells->speeds[5][idx_ws];
      tmp_cells->speeds[6][idx] = cells->speeds[6][idx_es];
      tmp_cells->speeds[7][idx] = cells->speeds[7][idx_en];
      tmp_cells->speeds[8][idx] = cells->speeds[8][idx_wn];
    }
  }

  /* ---- LOOP 2: REBOUND + COLLISION (tmp_cells -> cells) ---- */
  const float c_sq = 1.f / 3.f;
  const float w0 = 4.f / 9.f;
  const float w1 = 1.f / 9.f;
  const float w2 = 1.f / 36.f;

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      int idx = IDX(ii, jj, params.nx);

      if (obstacles[idx])
      {
        /* rebound: mirror propagated values back into cells */
        cells->speeds[0][idx] = tmp_cells->speeds[0][idx];
        cells->speeds[1][idx] = tmp_cells->speeds[3][idx];
        cells->speeds[2][idx] = tmp_cells->speeds[4][idx];
        cells->speeds[3][idx] = tmp_cells->speeds[1][idx];
        cells->speeds[4][idx] = tmp_cells->speeds[2][idx];
        cells->speeds[5][idx] = tmp_cells->speeds[7][idx];
        cells->speeds[6][idx] = tmp_cells->speeds[8][idx];
        cells->speeds[7][idx] = tmp_cells->speeds[5][idx];
        cells->speeds[8][idx] = tmp_cells->speeds[6][idx];
      }
      else
      {
        /* compute local density from propagated distributions */
        float t0 = tmp_cells->speeds[0][idx];
        float t1 = tmp_cells->speeds[1][idx];
        float t2 = tmp_cells->speeds[2][idx];
        float t3 = tmp_cells->speeds[3][idx];
        float t4 = tmp_cells->speeds[4][idx];
        float t5 = tmp_cells->speeds[5][idx];
        float t6 = tmp_cells->speeds[6][idx];
        float t7 = tmp_cells->speeds[7][idx];
        float t8 = tmp_cells->speeds[8][idx];

        float local_density = t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8;

        float u_x = (t1 + t5 + t8 - (t3 + t6 + t7)) / local_density;
        float u_y = (t2 + t5 + t6 - (t4 + t7 + t8)) / local_density;

        float u_sq = u_x*u_x + u_y*u_y;

        /* directional velocities */
        float u1 =  u_x;
        float u2 =  u_y;
        float u3 = -u_x;
        float u4 = -u_y;
        float u5 =  u_x + u_y;
        float u6 = -u_x + u_y;
        float u7 = -u_x - u_y;
        float u8 =  u_x - u_y;

        float inv_csq = 1.f / c_sq;
        float inv_csq2 = inv_csq * inv_csq;
        float half_inv_csq = 0.5f * inv_csq;

        /* equilibrium distributions */
        float d0 = w0 * local_density * (1.f - u_sq * half_inv_csq);

        float d1 = w1 * local_density * (1.f + u1*inv_csq + 0.5f*(u1*u1)*inv_csq2 - u_sq*half_inv_csq);
        float d2 = w1 * local_density * (1.f + u2*inv_csq + 0.5f*(u2*u2)*inv_csq2 - u_sq*half_inv_csq);
        float d3 = w1 * local_density * (1.f + u3*inv_csq + 0.5f*(u3*u3)*inv_csq2 - u_sq*half_inv_csq);
        float d4 = w1 * local_density * (1.f + u4*inv_csq + 0.5f*(u4*u4)*inv_csq2 - u_sq*half_inv_csq);

        float d5 = w2 * local_density * (1.f + u5*inv_csq + 0.5f*(u5*u5)*inv_csq2 - u_sq*half_inv_csq);
        float d6 = w2 * local_density * (1.f + u6*inv_csq + 0.5f*(u6*u6)*inv_csq2 - u_sq*half_inv_csq);
        float d7 = w2 * local_density * (1.f + u7*inv_csq + 0.5f*(u7*u7)*inv_csq2 - u_sq*half_inv_csq);
        float d8 = w2 * local_density * (1.f + u8*inv_csq + 0.5f*(u8*u8)*inv_csq2 - u_sq*half_inv_csq);

        /* relaxation step: write back into cells */
        float om = params.omega;

        cells->speeds[0][idx] = t0 + om * (d0 - t0);
        cells->speeds[1][idx] = t1 + om * (d1 - t1);
        cells->speeds[2][idx] = t2 + om * (d2 - t2);
        cells->speeds[3][idx] = t3 + om * (d3 - t3);
        cells->speeds[4][idx] = t4 + om * (d4 - t4);
        cells->speeds[5][idx] = t5 + om * (d5 - t5);
        cells->speeds[6][idx] = t6 + om * (d6 - t6);
        cells->speeds[7][idx] = t7 + om * (d7 - t7);
        cells->speeds[8][idx] = t8 + om * (d8 - t8);
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
