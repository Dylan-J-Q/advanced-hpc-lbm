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

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
int timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int accelerate_flow(const t_param params, t_speed* cells, int* obstacles);

int propagate_and_rebound(const t_param params,
              const float * restrict c0,
              const float * restrict c1,
              const float * restrict c2,
              const float * restrict c3,
              const float * restrict c4,
              const float * restrict c5,
              const float * restrict c6,
              const float * restrict c7,
              const float * restrict c8,
              float * restrict t0,
              float * restrict t1,
              float * restrict t2,
              float * restrict t3,
              float * restrict t4,
              float * restrict t5,
              float * restrict t6,
              float * restrict t7,
              float * restrict t8,
              const int   * restrict obstacles);

int collision(const t_param params,
              const float * restrict c0,
              const float * restrict c1,
              const float * restrict c2,
              const float * restrict c3,
              const float * restrict c4,
              const float * restrict c5,
              const float * restrict c6,
              const float * restrict c7,
              const float * restrict c8,
              float * restrict t0,
              float * restrict t1,
              float * restrict t2,
              float * restrict t3,
              float * restrict t4,
              float * restrict t5,
              float * restrict t6,
              float * restrict t7,
              float * restrict t8,
              const int * restrict obstacles);

int stream_collide(
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

/* compute average velocity */
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
              const int* obstacles);

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
    timestep(params, cells, tmp_cells, obstacles);
    av_vels[tt] = av_velocity(params,
          cells->speeds[0], cells->speeds[1], cells->speeds[2],
          cells->speeds[3], cells->speeds[4], cells->speeds[5],
          cells->speeds[6], cells->speeds[7], cells->speeds[8],
          obstacles);
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
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
  printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
  printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
  printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
  printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);
  write_values(params, cells, obstacles, av_vels);
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);

  return EXIT_SUCCESS;
}

int timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles)
{
    accelerate_flow(params, cells, obstacles);

    // stream + bounce-back + collide: read cells, write tmp
    stream_collide(params,
        cells->speeds[0], cells->speeds[1], cells->speeds[2],
        cells->speeds[3], cells->speeds[4], cells->speeds[5],
        cells->speeds[6], cells->speeds[7], cells->speeds[8],
        tmp_cells->speeds[0], tmp_cells->speeds[1], tmp_cells->speeds[2],
        tmp_cells->speeds[3], tmp_cells->speeds[4], tmp_cells->speeds[5],
        tmp_cells->speeds[6], tmp_cells->speeds[7], tmp_cells->speeds[8],
        obstacles
    );

    // swap pointers like before
    t_speed temp = *cells;
    *cells = *tmp_cells;
    *tmp_cells = temp;

    return EXIT_SUCCESS;
}


int accelerate_flow(const t_param params, t_speed* restrict cells, int* restrict obstacles)
{
  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  int jj = params.ny - 2;
  int row = jj * params.nx;

  for (int ii = 0; ii < params.nx; ii++)
  {
    int idx = ii + row;
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii + jj*params.nx]
        && (cells->speeds[3][idx] - w1) > 0.f
        && (cells->speeds[6][idx] - w2) > 0.f
        && (cells->speeds[7][idx] - w2) > 0.f)
    {
      /* increase 'east-side' densities */
      cells->speeds[1][idx] += w1;
      cells->speeds[5][idx] += w2;
      cells->speeds[8][idx] += w2;
      /* decrease 'west-side' densities */
      cells->speeds[3][idx] -= w1;
      cells->speeds[6][idx] -= w2;
      cells->speeds[7][idx] -= w2;
    }
  }

  return EXIT_SUCCESS;
}

#include <math.h>
#include <stdlib.h>

static inline void collide_or_copy(
    int is_obstacle,
    float omega,
    float s0,float s1,float s2,float s3,float s4,float s5,float s6,float s7,float s8,
    float *o0,float *o1,float *o2,float *o3,float *o4,float *o5,float *o6,float *o7,float *o8
){
    if (is_obstacle) {
        // obstacles: do NOT collide, just keep post-bounce-back populations
        *o0 = s0; *o1 = s1; *o2 = s2; *o3 = s3; *o4 = s4;
        *o5 = s5; *o6 = s6; *o7 = s7; *o8 = s8;
        return;
    }

    // BGK collision (same math as your collision)
    const float w0 = 4.f/9.f, w1 = 1.f/9.f, w2 = 1.f/36.f;
    const float inv_c_sq = 3.f;
    const float inv_2_c_sq = 1.5f;
    const float inv_2_c_sq_sq = 4.5f;

    float rho = s0+s1+s2+s3+s4+s5+s6+s7+s8;

    // rho should be > 0 for fluid; if it ever goes tiny due to a bug, avoid NaNs
    if (rho <= 1e-20f) rho = 1e-20f;

    float inv_rho = 1.f / rho;

    float ux = (s1+s5+s8 - (s3+s6+s7)) * inv_rho;
    float uy = (s2+s5+s6 - (s4+s7+s8)) * inv_rho;

    float u2 = ux*ux + uy*uy;
    float common = 1.f - inv_2_c_sq * u2;

    float feq0 = w0 * rho * common;
    float feq1 = w1 * rho * (common + inv_c_sq*ux + inv_2_c_sq_sq*ux*ux);
    float feq2 = w1 * rho * (common + inv_c_sq*uy + inv_2_c_sq_sq*uy*uy);
    float feq3 = w1 * rho * (common - inv_c_sq*ux + inv_2_c_sq_sq*ux*ux);
    float feq4 = w1 * rho * (common - inv_c_sq*uy + inv_2_c_sq_sq*uy*uy);

    float uxy = ux + uy;
    float feq5 = w2 * rho * (common + inv_c_sq*uxy + inv_2_c_sq_sq*uxy*uxy);
    uxy = -ux + uy;
    float feq6 = w2 * rho * (common + inv_c_sq*uxy + inv_2_c_sq_sq*uxy*uxy);
    uxy = -ux - uy;
    float feq7 = w2 * rho * (common + inv_c_sq*uxy + inv_2_c_sq_sq*uxy*uxy);
    uxy = ux - uy;
    float feq8 = w2 * rho * (common + inv_c_sq*uxy + inv_2_c_sq_sq*uxy*uxy);

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

int stream_collide(
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
    const int nx = params.nx;
    const int ny = params.ny;
    const float omega = params.omega;

    // ----------------------
    // Interior (vectorisable)
    // ----------------------
    for (int jj = 1; jj < ny - 1; ++jj) {
        const int row       = jj * nx;
        const int row_north = (jj + 1) * nx;
        const int row_south = (jj - 1) * nx;

        #pragma omp simd
        for (int ii = 1; ii < nx - 1; ++ii) {
            const int idx = row + ii;

            // stream (gather from neighbours)
            float r0 = c0[idx];
            float r1 = c1[idx - 1];
            float r2 = c2[row_south + ii];
            float r3 = c3[idx + 1];
            float r4 = c4[row_north + ii];
            float r5 = c5[row_south + ii - 1];
            float r6 = c6[row_south + ii + 1];
            float r7 = c7[row_north + ii + 1];
            float r8 = c8[row_north + ii - 1];

            // bounce-back if obstacle at destination cell
            const int obs = obstacles[idx];
            float s0 = r0;
            float s1 = obs ? r3 : r1;
            float s2 = obs ? r4 : r2;
            float s3 = obs ? r1 : r3;
            float s4 = obs ? r2 : r4;
            float s5 = obs ? r7 : r5;
            float s6 = obs ? r8 : r6;
            float s7 = obs ? r5 : r7;
            float s8 = obs ? r6 : r8;

            // collide (or copy for obstacles)
            collide_or_copy(obs, omega, s0,s1,s2,s3,s4,s5,s6,s7,s8,
                            &out0[idx],&out1[idx],&out2[idx],&out3[idx],&out4[idx],
                            &out5[idx],&out6[idx],&out7[idx],&out8[idx]);
        }
    }

    // -------------
    // Top row jj = 0
    // -------------
    {
        const int row       = 0;
        const int row_north = nx;
        const int row_south = (ny - 1) * nx;

        for (int ii = 0; ii < nx; ++ii) {
            const int ii_w = (ii == 0)    ? nx - 1 : ii - 1;
            const int ii_e = (ii == nx-1) ? 0      : ii + 1;
            const int idx  = row + ii;

            float r0 = c0[idx];
            float r1 = c1[row + ii_w];
            float r2 = c2[row_south + ii];
            float r3 = c3[row + ii_e];
            float r4 = c4[row_north + ii];
            float r5 = c5[row_south + ii_w];
            float r6 = c6[row_south + ii_e];
            float r7 = c7[row_north + ii_e];
            float r8 = c8[row_north + ii_w];

            const int obs = obstacles[idx];
            float s0 = r0;
            float s1 = obs ? r3 : r1;
            float s2 = obs ? r4 : r2;
            float s3 = obs ? r1 : r3;
            float s4 = obs ? r2 : r4;
            float s5 = obs ? r7 : r5;
            float s6 = obs ? r8 : r6;
            float s7 = obs ? r5 : r7;
            float s8 = obs ? r6 : r8;

            collide_or_copy(obs, omega, s0,s1,s2,s3,s4,s5,s6,s7,s8,
                            &out0[idx],&out1[idx],&out2[idx],&out3[idx],&out4[idx],
                            &out5[idx],&out6[idx],&out7[idx],&out8[idx]);
        }
    }

    // ----------------
    // Bottom row jj=ny-1
    // ----------------
    {
        const int jj = ny - 1;
        const int row       = jj * nx;
        const int row_north = 0;
        const int row_south = (jj - 1) * nx;

        for (int ii = 0; ii < nx; ++ii) {
            const int ii_w = (ii == 0)    ? nx - 1 : ii - 1;
            const int ii_e = (ii == nx-1) ? 0      : ii + 1;
            const int idx  = row + ii;

            float r0 = c0[idx];
            float r1 = c1[row + ii_w];
            float r2 = c2[row_south + ii];
            float r3 = c3[row + ii_e];
            float r4 = c4[row_north + ii];
            float r5 = c5[row_south + ii_w];
            float r6 = c6[row_south + ii_e];
            float r7 = c7[row_north + ii_e];
            float r8 = c8[row_north + ii_w];

            const int obs = obstacles[idx];
            float s0 = r0;
            float s1 = obs ? r3 : r1;
            float s2 = obs ? r4 : r2;
            float s3 = obs ? r1 : r3;
            float s4 = obs ? r2 : r4;
            float s5 = obs ? r7 : r5;
            float s6 = obs ? r8 : r6;
            float s7 = obs ? r5 : r7;
            float s8 = obs ? r6 : r8;

            collide_or_copy(obs, omega, s0,s1,s2,s3,s4,s5,s6,s7,s8,
                            &out0[idx],&out1[idx],&out2[idx],&out3[idx],&out4[idx],
                            &out5[idx],&out6[idx],&out7[idx],&out8[idx]);
        }
    }

    // ------------------------
    // Left + right columns
    // ------------------------
    for (int jj = 1; jj < ny - 1; ++jj) {
        const int row       = jj * nx;
        const int row_north = (jj + 1) * nx;
        const int row_south = (jj - 1) * nx;

        // left column ii=0
        {
            const int idx  = row;
            const int ii_w = nx - 1;
            const int ii_e = 1;

            float r0 = c0[idx];
            float r1 = c1[row + ii_w];
            float r2 = c2[row_south];
            float r3 = c3[row + ii_e];
            float r4 = c4[row_north];
            float r5 = c5[row_south + ii_w];
            float r6 = c6[row_south + ii_e];
            float r7 = c7[row_north + ii_e];
            float r8 = c8[row_north + ii_w];

            const int obs = obstacles[idx];
            float s0 = r0;
            float s1 = obs ? r3 : r1;
            float s2 = obs ? r4 : r2;
            float s3 = obs ? r1 : r3;
            float s4 = obs ? r2 : r4;
            float s5 = obs ? r7 : r5;
            float s6 = obs ? r8 : r6;
            float s7 = obs ? r5 : r7;
            float s8 = obs ? r6 : r8;

            collide_or_copy(obs, omega, s0,s1,s2,s3,s4,s5,s6,s7,s8,
                            &out0[idx],&out1[idx],&out2[idx],&out3[idx],&out4[idx],
                            &out5[idx],&out6[idx],&out7[idx],&out8[idx]);
        }

        // right column ii=nx-1
        {
            const int ii  = nx - 1;
            const int idx = row + ii;
            const int ii_w = ii - 1;
            const int ii_e = 0;

            float r0 = c0[idx];
            float r1 = c1[row + ii_w];
            float r2 = c2[row_south + ii];
            float r3 = c3[row + ii_e];
            float r4 = c4[row_north + ii];
            float r5 = c5[row_south + ii_w];
            float r6 = c6[row_south + ii_e];
            float r7 = c7[row_north + ii_e];
            float r8 = c8[row_north + ii_w];

            const int obs = obstacles[idx];
            float s0 = r0;
            float s1 = obs ? r3 : r1;
            float s2 = obs ? r4 : r2;
            float s3 = obs ? r1 : r3;
            float s4 = obs ? r2 : r4;
            float s5 = obs ? r7 : r5;
            float s6 = obs ? r8 : r6;
            float s7 = obs ? r5 : r7;
            float s8 = obs ? r6 : r8;

            collide_or_copy(obs, omega, s0,s1,s2,s3,s4,s5,s6,s7,s8,
                            &out0[idx],&out1[idx],&out2[idx],&out3[idx],&out4[idx],
                            &out5[idx],&out6[idx],&out7[idx],&out8[idx]);
        }
    }

    return EXIT_SUCCESS;
}


int propagate_and_rebound(const t_param params,
              const float * restrict c0,
              const float * restrict c1,
              const float * restrict c2,
              const float * restrict c3,
              const float * restrict c4,
              const float * restrict c5,
              const float * restrict c6,
              const float * restrict c7,
              const float * restrict c8,
              float * restrict t0,
              float * restrict t1,
              float * restrict t2,
              float * restrict t3,
              float * restrict t4,
              float * restrict t5,
              float * restrict t6,
              float * restrict t7,
              float * restrict t8,
              const int   * restrict obstacles)
{
    const int nx = params.nx;
    const int ny = params.ny;

    //interior
    for (int jj = 1; jj < ny - 1; ++jj)
    {
        const int row       = jj * nx;
        const int row_north = (jj + 1) * nx;
        const int row_south = (jj - 1) * nx;

        #pragma omp simd
        for (int ii = 1; ii < nx - 1; ++ii)
        {
            const int idx = row + ii;

            const float s0 = c0[idx];

            const float s1 = c1[idx - 1];
            const float s2 = c2[row_south + ii];
            const float s3 = c3[idx + 1];
            const float s4 = c4[row_north + ii]; 

            const float s5 = c5[row_south + ii - 1];
            const float s6 = c6[row_south + ii + 1];
            const float s7 = c7[row_north + ii + 1];
            const float s8 = c8[row_north + ii - 1];

            const float m  = (float)obstacles[idx];
            const float nm = 1.0f - m;

            t0[idx] = s0;
            t1[idx] = m * s3 + nm * s1;
            t2[idx] = m * s4 + nm * s2;
            t3[idx] = m * s1 + nm * s3;
            t4[idx] = m * s2 + nm * s4;
            t5[idx] = m * s7 + nm * s5;
            t6[idx] = m * s8 + nm * s6;
            t7[idx] = m * s5 + nm * s7;
            t8[idx] = m * s6 + nm * s8;
        }
    }

    //top row
    {
        const int row       = 0;
        const int row_north = nx;
        const int row_south = (ny - 1) * nx;

        for (int ii = 0; ii < nx; ++ii)
        {
            const int ii_w = (ii == 0)    ? nx - 1 : ii - 1;
            const int ii_e = (ii == nx-1) ? 0      : ii + 1;
            const int idx  = row + ii;

            const float s0 = c0[idx];

            const float s1 = c1[row + ii_w];
            const float s2 = c2[row_south + ii];
            const float s3 = c3[row + ii_e];
            const float s4 = c4[row_north + ii];

            const float s5 = c5[row_south + ii_w];
            const float s6 = c6[row_south + ii_e];
            const float s7 = c7[row_north + ii_e];
            const float s8 = c8[row_north + ii_w];

            const float m  = (float)obstacles[idx];
            const float nm = 1.0f - m;

            t0[idx] = s0;
            t1[idx] = m * s3 + nm * s1;
            t2[idx] = m * s4 + nm * s2;
            t3[idx] = m * s1 + nm * s3;
            t4[idx] = m * s2 + nm * s4;
            t5[idx] = m * s7 + nm * s5;
            t6[idx] = m * s8 + nm * s6;
            t7[idx] = m * s5 + nm * s7;
            t8[idx] = m * s6 + nm * s8;
        }
    }

    //bottom row
    {
        const int jj = ny - 1;
        const int row = jj * nx;
        const int row_north = 0;
        const int row_south = (jj - 1) * nx;

        for (int ii = 0; ii < nx; ++ii)
        {
            const int ii_w = (ii == 0)    ? nx - 1 : ii - 1;
            const int ii_e = (ii == nx-1) ? 0      : ii + 1;
            const int idx  = row + ii;

            const float s0 = c0[idx];

            const float s1 = c1[row + ii_w];
            const float s2 = c2[row_south + ii];
            const float s3 = c3[row + ii_e];
            const float s4 = c4[row_north + ii];

            const float s5 = c5[row_south + ii_w];
            const float s6 = c6[row_south + ii_e];
            const float s7 = c7[row_north + ii_e];
            const float s8 = c8[row_north + ii_w];

            const float m  = (float)obstacles[idx];
            const float nm = 1.0f - m;

            t0[idx] = s0;
            t1[idx] = m * s3 + nm * s1;
            t2[idx] = m * s4 + nm * s2;
            t3[idx] = m * s1 + nm * s3;
            t4[idx] = m * s2 + nm * s4;
            t5[idx] = m * s7 + nm * s5;
            t6[idx] = m * s8 + nm * s6;
            t7[idx] = m * s5 + nm * s7;
            t8[idx] = m * s6 + nm * s8;
        }
    }

    for (int jj = 1; jj < ny - 1; ++jj)
    {
        const int row       = jj * nx;
        const int row_north = (jj + 1) * nx;
        const int row_south = (jj - 1) * nx;

        //left column
        {
            const int idx = row;
            const int ii_w = nx - 1;
            const int ii_e = 1;

            const float s0 = c0[idx];

            const float s1 = c1[row + ii_w];
            const float s2 = c2[row_south];
            const float s3 = c3[row + ii_e];
            const float s4 = c4[row_north];

            const float s5 = c5[row_south + ii_w];
            const float s6 = c6[row_south + ii_e];
            const float s7 = c7[row_north + ii_e];
            const float s8 = c8[row_north + ii_w];

            const float m  = (float)obstacles[idx];
            const float nm = 1.0f - m;

            t0[idx] = s0;
            t1[idx] = m * s3 + nm * s1;
            t2[idx] = m * s4 + nm * s2;
            t3[idx] = m * s1 + nm * s3;
            t4[idx] = m * s2 + nm * s4;
            t5[idx] = m * s7 + nm * s5;
            t6[idx] = m * s8 + nm * s6;
            t7[idx] = m * s5 + nm * s7;
            t8[idx] = m * s6 + nm * s8;
        }

        //right column
        {
            const int ii = nx - 1;
            const int idx = row + ii;
            const int ii_w = ii - 1;
            const int ii_e = 0;

            const float s0 = c0[idx];

            const float s1 = c1[row + ii_w];
            const float s2 = c2[row_south + ii];
            const float s3 = c3[row + ii_e];
            const float s4 = c4[row_north + ii];

            const float s5 = c5[row_south + ii_w];
            const float s6 = c6[row_south + ii_e];
            const float s7 = c7[row_north + ii_e];
            const float s8 = c8[row_north + ii_w];

            const float m  = (float)obstacles[idx];
            const float nm = 1.0f - m;

            t0[idx] = s0;
            t1[idx] = m * s3 + nm * s1;
            t2[idx] = m * s4 + nm * s2;
            t3[idx] = m * s1 + nm * s3;
            t4[idx] = m * s2 + nm * s4;
            t5[idx] = m * s7 + nm * s5;
            t6[idx] = m * s8 + nm * s6;
            t7[idx] = m * s5 + nm * s7;
            t8[idx] = m * s6 + nm * s8;
        }
    }
    return EXIT_SUCCESS;
}

int collision(const t_param params,
              const float * restrict c0,
              const float * restrict c1,
              const float * restrict c2,
              const float * restrict c3,
              const float * restrict c4,
              const float * restrict c5,
              const float * restrict c6,
              const float * restrict c7,
              const float * restrict c8,
              float * restrict t0,
              float * restrict t1,
              float * restrict t2,
              float * restrict t3,
              float * restrict t4,
              float * restrict t5,
              float * restrict t6,
              float * restrict t7,
              float * restrict t8,
              const int * restrict obstacles)
{
  const int N = params.nx * params.ny;
  const float omega = params.omega;

  const float w0 = 4.f/9.f, w1 = 1.f/9.f, w2 = 1.f/36.f;
  const float inv_c_sq = 3.f;
  const float inv_2_c_sq = 1.5f;
  const float inv_2_c_sq_sq = 4.5f;

  #pragma omp simd aligned(c0,c1,c2,c3,c4,c5,c6,c7,c8, \
                           t0,t1,t2,t3,t4,t5,t6,t7,t8,obstacles:32)
  for (int idx = 0; idx < N; ++idx)
  {
    float mask = (float)(!obstacles[idx]);

    float s0 = c0[idx];
    float s1 = c1[idx];
    float s2 = c2[idx];
    float s3 = c3[idx];
    float s4 = c4[idx];
    float s5 = c5[idx];
    float s6 = c6[idx];
    float s7 = c7[idx];
    float s8 = c8[idx];

    float rho = s0+s1+s2+s3+s4+s5+s6+s7+s8;
    rho = mask * rho + (1.f - mask);

    float inv_rho = 1.f / rho;

    float ux = (s1+s5+s8 - (s3+s6+s7)) * inv_rho;
    float uy = (s2+s5+s6 - (s4+s7+s8)) * inv_rho;

    float u2 = ux*ux + uy*uy;
    float common = 1.f - inv_2_c_sq * u2;

    float feq0 = w0 * rho * common;
    float feq1 = w1 * rho * (common + inv_c_sq*ux + inv_2_c_sq_sq*ux*ux);
    float feq2 = w1 * rho * (common + inv_c_sq*uy + inv_2_c_sq_sq*uy*uy);
    float feq3 = w1 * rho * (common - inv_c_sq*ux + inv_2_c_sq_sq*ux*ux);
    float feq4 = w1 * rho * (common - inv_c_sq*uy + inv_2_c_sq_sq*uy*uy);

    float uxy = ux + uy;
    float feq5 = w2 * rho * (common + inv_c_sq*uxy + inv_2_c_sq_sq*uxy*uxy);
    uxy = -ux + uy;
    float feq6 = w2 * rho * (common + inv_c_sq*uxy + inv_2_c_sq_sq*uxy*uxy);
    uxy = -ux - uy;
    float feq7 = w2 * rho * (common + inv_c_sq*uxy + inv_2_c_sq_sq*uxy*uxy);
    uxy = ux - uy;
    float feq8 = w2 * rho * (common + inv_c_sq*uxy + inv_2_c_sq_sq*uxy*uxy);

    t0[idx] = s0 + mask * omega * (feq0 - s0);
    t1[idx] = s1 + mask * omega * (feq1 - s1);
    t2[idx] = s2 + mask * omega * (feq2 - s2);
    t3[idx] = s3 + mask * omega * (feq3 - s3);
    t4[idx] = s4 + mask * omega * (feq4 - s4);
    t5[idx] = s5 + mask * omega * (feq5 - s5);
    t6[idx] = s6 + mask * omega * (feq6 - s6);
    t7[idx] = s7 + mask * omega * (feq7 - s7);
    t8[idx] = s8 + mask * omega * (feq8 - s8);
  }

  return EXIT_SUCCESS;
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
    const int nx = params.nx;
    const int ny = params.ny;
    const int n  = nx * ny;

    float tot_u = 0.f;
    int tot_cells = 0;

    for (int idx = 0; idx < n; ++idx)
    {
        // Skip obstacles early
        if (obstacles[idx]) continue;

        // Load speeds
        float s0 = c0[idx];
        float s1 = c1[idx];
        float s2 = c2[idx];
        float s3 = c3[idx];
        float s4 = c4[idx];
        float s5 = c5[idx];
        float s6 = c6[idx];
        float s7 = c7[idx];
        float s8 = c8[idx];

        // Compute local density once
        float rho = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8;
        float inv_rho = 1.f / rho;

        // Velocity components
        float ux = (s1 + s5 + s8 - (s3 + s6 + s7)) * inv_rho;
        float uy = (s2 + s5 + s6 - (s4 + s7 + s8)) * inv_rho;

        // Accumulate magnitude
        tot_u += sqrtf(ux*ux + uy*uy);
        tot_cells++;
    }

    // Avoid division by zero
    return (tot_cells > 0) ? tot_u / (float)tot_cells : 0.f;
}



int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
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
  //*cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));
  int n_cells = params->nx * params->ny;

  t_speed *cells = malloc(sizeof(t_speed));
  *cells_ptr = cells;

  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  //*tmp_cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));
  t_speed *tmp_cells = malloc(sizeof(t_speed));
  *tmp_cells_ptr = tmp_cells;

  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  
  for(int k =0; k < NSPEEDS; k++){
    cells->speeds[k] = malloc(n_cells * sizeof(float));
    tmp_cells->speeds[k] = malloc(n_cells * sizeof(float));
    
    if (!cells->speeds[k] || !tmp_cells->speeds[k]) {
      die("cannot allocate speeds", __LINE__, __FILE__);
    }
  }

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

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

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr)
{
  /*
  ** free up allocated memory
  */
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
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        total += cells->speeds[kk][ii + jj*params.nx];
      }
    }
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
        local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells->speeds[kk][ii + jj*params.nx];
        }

        /* compute x velocity component */
        u_x = (cells->speeds[1][ii + jj*params.nx]
               + cells->speeds[5][ii + jj*params.nx]
               + cells->speeds[8][ii + jj*params.nx]
               - (cells->speeds[3][ii + jj*params.nx]
                  + cells->speeds[6][ii + jj*params.nx]
                  + cells->speeds[7][ii + jj*params.nx]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells->speeds[2][ii + jj*params.nx]
               + cells->speeds[5][ii + jj*params.nx]
               + cells->speeds[6][ii + jj*params.nx]
               - (cells->speeds[4][ii + jj*params.nx]
                  + cells->speeds[7][ii + jj*params.nx]
                  + cells->speeds[8][ii + jj*params.nx]))
              / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
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
