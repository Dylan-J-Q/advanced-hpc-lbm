/*
** MPI-parallel d2q9-bgk lattice Boltzmann code.
**
** 1D row decomposition: the global ny rows are split across ranks.
** Each rank owns local_ny rows and maintains one halo (ghost) row
** above and below for streaming across rank boundaries.
**
** Memory layout per rank (each speed array):
**
**   row 0                 = south halo  (copy of neighbour's top real row)
**   rows 1 .. local_ny    = real rows   (this rank's data)
**   row local_ny + 1      = north halo  (copy of neighbour's bottom real row)
**
** The 'speeds' in each cell are numbered as follows:
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
#include <sys/time.h>
#include <mpi.h>

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

/* ------------------------------------------------------------------ */
/*  Data structures                                                    */
/* ------------------------------------------------------------------ */

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

typedef struct {
  int rank;
  int nprocs;
  int local_ny;
  int jj_start;
  int rank_south;
  int rank_north;
  int alloc_rows;
} t_decomp;

/* ------------------------------------------------------------------ */
/*  Prototypes                                                         */
/* ------------------------------------------------------------------ */

void compute_decomposition(int rank, int nprocs, int ny, t_decomp *d);

void initialise(const char *paramfile, const char *obstaclefile,
                t_param *params, t_decomp *decomp,
                t_speed **cells_ptr, t_speed **tmp_cells_ptr,
                int **obstacles_ptr, float **av_vels_ptr);

void finalise(t_speed **cells_ptr, t_speed **tmp_cells_ptr,
              int **obstacles_ptr, float **av_vels_ptr);

void halo_exchange(t_speed *cells, const t_decomp *d, int nx);

void accelerate_flow(const t_param params, const t_decomp *d,
                     t_speed *cells, const int *obstacles);

void propagate_rebound_collide(
    const t_param params, const t_decomp *d,
    const t_speed *cells, t_speed *tmp_cells, const int *obstacles,
    float *local_tot_u, int *local_tot_cells);

void gather_and_write(const t_param params, const t_decomp *d,
                      const t_speed *cells, const int *obstacles,
                      const float *av_vels);

float calc_reynolds(const t_param params, const t_decomp *d,
                    const t_speed *cells, const int *obstacles);

void die(const char *msg, const int line, const char *file);

/* ------------------------------------------------------------------ */
/*  Collision kernel (unchanged from serial)                           */
/* ------------------------------------------------------------------ */

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
  float feq1 = w1 * rho * (common + 3.f * ux  + 4.5f * ux * ux);
  float feq2 = w1 * rho * (common + 3.f * uy  + 4.5f * uy * uy);
  float feq3 = w1 * rho * (common - 3.f * ux  + 4.5f * ux * ux);
  float feq4 = w1 * rho * (common - 3.f * uy  + 4.5f * uy * uy);

  float uxy;
  uxy =  ux + uy; float feq5 = w2 * rho * (common + 3.f*uxy + 4.5f*uxy*uxy);
  uxy = -ux + uy; float feq6 = w2 * rho * (common + 3.f*uxy + 4.5f*uxy*uxy);
  uxy = -ux - uy; float feq7 = w2 * rho * (common + 3.f*uxy + 4.5f*uxy*uxy);
  uxy =  ux - uy; float feq8 = w2 * rho * (common + 3.f*uxy + 4.5f*uxy*uxy);

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

/* ================================================================== */
/*  main                                                               */
/* ================================================================== */

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);

  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if (argc != 3) {
    if (rank == 0) fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", argv[0]);
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  t_param  params;
  t_decomp decomp;
  t_speed *cells     = NULL;
  t_speed *tmp_cells = NULL;
  int     *obstacles = NULL;
  float   *av_vels   = NULL;

  struct timeval timstr;
  gettimeofday(&timstr, NULL);
  double tot_tic  = timstr.tv_sec + timstr.tv_usec / 1e6;
  double init_tic = tot_tic;

  initialise(argv[1], argv[2], &params, &decomp,
             &cells, &tmp_cells, &obstacles, &av_vels);

  gettimeofday(&timstr, NULL);
  double init_toc = timstr.tv_sec + timstr.tv_usec / 1e6;
  double comp_tic = init_toc;

  /* ---- timestep loop ---- */
  for (int tt = 0; tt < params.maxIters; tt++)
  {
    accelerate_flow(params, &decomp, cells, obstacles);

    halo_exchange(cells, &decomp, params.nx);

    float local_tot_u    = 0.f;
    int   local_tot_cells = 0;
    propagate_rebound_collide(params, &decomp, cells, tmp_cells,
                              obstacles, &local_tot_u, &local_tot_cells);

    /* reduce partial sums to get global average velocity */
    float global_tot_u;
    int   global_tot_cells;
    MPI_Allreduce(&local_tot_u,    &global_tot_u,    1,
                  MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_tot_cells,&global_tot_cells, 1,
                  MPI_INT,   MPI_SUM, MPI_COMM_WORLD);

    av_vels[tt] = (global_tot_cells > 0)
                ? global_tot_u / (float)global_tot_cells
                : 0.f;

    /* swap grids */
    t_speed temp = *cells;
    *cells       = *tmp_cells;
    *tmp_cells   = temp;

#ifdef DEBUG
    if (rank == 0) {
      printf("==timestep: %d==\n", tt);
      printf("av velocity: %.12E\n", av_vels[tt]);
    }
#endif
  }

  gettimeofday(&timstr, NULL);
  double comp_toc = timstr.tv_sec + timstr.tv_usec / 1e6;
  double col_tic  = comp_toc;

  gather_and_write(params, &decomp, cells, obstacles, av_vels);

  gettimeofday(&timstr, NULL);
  double col_toc = timstr.tv_sec + timstr.tv_usec / 1e6;
  double tot_toc = col_toc;

  if (rank == 0) {
    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n",
           calc_reynolds(params, &decomp, cells, obstacles));
    printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
    printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
    printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
    printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);
  }

  finalise(&cells, &tmp_cells, &obstacles, &av_vels);
  MPI_Finalize();
  return EXIT_SUCCESS;
}

/* ================================================================== */
/*  Domain decomposition                                               */
/* ================================================================== */

void compute_decomposition(int rank, int nprocs, int ny, t_decomp *d)
{
  d->rank    = rank;
  d->nprocs  = nprocs;

  int base     = ny / nprocs;
  int remainder = ny % nprocs;

  d->local_ny = base + (rank < remainder ? 1 : 0);
  d->jj_start = rank * base + (rank < remainder ? rank : remainder);

  d->rank_south = (rank == 0)          ? nprocs - 1 : rank - 1;
  d->rank_north = (rank == nprocs - 1) ? 0          : rank + 1;

  d->alloc_rows = d->local_ny + 2;
}

/* ================================================================== */
/*  Halo exchange (non-blocking for all 9 speeds)                      */
/* ================================================================== */

void halo_exchange(t_speed *cells, const t_decomp *d, int nx)
{
  const int local_ny = d->local_ny;

  MPI_Request reqs[4 * NSPEEDS];
  int nreqs = 0;

  for (int k = 0; k < NSPEEDS; k++) {
    float *sp = cells->speeds[k];

    /* bottom real row (1) -> south neighbour,
       receive south halo (0) <- south neighbour */
    MPI_Isend(&sp[1 * nx],              nx, MPI_FLOAT,
              d->rank_south, k,           MPI_COMM_WORLD, &reqs[nreqs++]);
    MPI_Irecv(&sp[0],                   nx, MPI_FLOAT,
              d->rank_south, k + NSPEEDS, MPI_COMM_WORLD, &reqs[nreqs++]);

    /* top real row (local_ny) -> north neighbour,
       receive north halo (local_ny+1) <- north neighbour */
    MPI_Isend(&sp[local_ny * nx],       nx, MPI_FLOAT,
              d->rank_north, k + NSPEEDS, MPI_COMM_WORLD, &reqs[nreqs++]);
    MPI_Irecv(&sp[(local_ny + 1) * nx], nx, MPI_FLOAT,
              d->rank_north, k,           MPI_COMM_WORLD, &reqs[nreqs++]);
  }

  MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);
}

/* ================================================================== */
/*  Accelerate flow (only the rank owning global row ny-2)             */
/* ================================================================== */

void accelerate_flow(const t_param params, const t_decomp *d,
                     t_speed *cells, const int *obstacles)
{
  const int nx         = params.nx;
  const int global_row = params.ny - 2;

  if (global_row < d->jj_start ||
      global_row >= d->jj_start + d->local_ny)
    return;

  const int local_jj = global_row - d->jj_start + 1;
  const int row      = local_jj * nx;

  const float w1 = params.density * params.accel / 9.f;
  const float w2 = params.density * params.accel / 36.f;

  for (int ii = 0; ii < nx; ii++)
  {
    const int idx = row + ii;

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
}

/* ================================================================== */
/*  Fused propagate + rebound + collide  (single pass, local grid)     */
/* ================================================================== */

void propagate_rebound_collide(
    const t_param params, const t_decomp *d,
    const t_speed *cells, t_speed *tmp_cells, const int *obstacles,
    float *local_tot_u, int *local_tot_cells)
{
  const int   nx       = params.nx;
  const int   local_ny = d->local_ny;
  const float omega    = params.omega;

  const float *c0 = cells->speeds[0], *c1 = cells->speeds[1];
  const float *c2 = cells->speeds[2], *c3 = cells->speeds[3];
  const float *c4 = cells->speeds[4], *c5 = cells->speeds[5];
  const float *c6 = cells->speeds[6], *c7 = cells->speeds[7];
  const float *c8 = cells->speeds[8];

  float *out0 = tmp_cells->speeds[0], *out1 = tmp_cells->speeds[1];
  float *out2 = tmp_cells->speeds[2], *out3 = tmp_cells->speeds[3];
  float *out4 = tmp_cells->speeds[4], *out5 = tmp_cells->speeds[5];
  float *out6 = tmp_cells->speeds[6], *out7 = tmp_cells->speeds[7];
  float *out8 = tmp_cells->speeds[8];

  float tot_u = 0.f;
  int   tot_c = 0;

  for (int jj = 1; jj <= local_ny; ++jj)
  {
    const int row   = jj * nx;
    const int row_s = (jj - 1) * nx;
    const int row_n = (jj + 1) * nx;

    for (int ii = 0; ii < nx; ++ii)
    {
      const int ii_w = (ii == 0) ? nx - 1 : ii - 1;
      const int ii_e = (ii == nx - 1) ? 0 : ii + 1;
      const int idx  = row + ii;

      /* stream: gather from neighbours */
      const float r0 = c0[idx];
      const float r1 = c1[row   + ii_w];
      const float r2 = c2[row_s + ii];
      const float r3 = c3[row   + ii_e];
      const float r4 = c4[row_n + ii];
      const float r5 = c5[row_s + ii_w];
      const float r6 = c6[row_s + ii_e];
      const float r7 = c7[row_n + ii_e];
      const float r8 = c8[row_n + ii_w];

      const int obs = obstacles[idx];

      if (obs) {
        out0[idx] = r0;
        out1[idx] = r3;  out2[idx] = r4;
        out3[idx] = r1;  out4[idx] = r2;
        out5[idx] = r7;  out6[idx] = r8;
        out7[idx] = r5;  out8[idx] = r6;
      } else {
        collide(omega, r0,r1,r2,r3,r4,r5,r6,r7,r8,
                &out0[idx],&out1[idx],&out2[idx],&out3[idx],&out4[idx],
                &out5[idx],&out6[idx],&out7[idx],&out8[idx]);

        const float rho     = r0+r1+r2+r3+r4+r5+r6+r7+r8;
        const float inv_rho = 1.f / rho;
        const float ux = (r1+r5+r8 - (r3+r6+r7)) * inv_rho;
        const float uy = (r2+r5+r6 - (r4+r7+r8)) * inv_rho;
        tot_u += sqrtf(ux*ux + uy*uy);
        tot_c++;
      }
    }
  }

  *local_tot_u     = tot_u;
  *local_tot_cells = tot_c;
}

/* ================================================================== */
/*  Initialise                                                         */
/* ================================================================== */

void initialise(const char *paramfile, const char *obstaclefile,
                t_param *params, t_decomp *decomp,
                t_speed **cells_ptr, t_speed **tmp_cells_ptr,
                int **obstacles_ptr, float **av_vels_ptr)
{
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  /* ---- rank 0 reads parameters ---- */
  if (rank == 0) {
    FILE *fp = fopen(paramfile, "r");
    if (!fp) die("could not open parameter file", __LINE__, __FILE__);

    if (fscanf(fp, "%d\n", &params->nx)           != 1) die("bad nx",          __LINE__, __FILE__);
    if (fscanf(fp, "%d\n", &params->ny)           != 1) die("bad ny",          __LINE__, __FILE__);
    if (fscanf(fp, "%d\n", &params->maxIters)     != 1) die("bad maxIters",    __LINE__, __FILE__);
    if (fscanf(fp, "%d\n", &params->reynolds_dim) != 1) die("bad reynolds_dim",__LINE__, __FILE__);
    if (fscanf(fp, "%f\n", &params->density)      != 1) die("bad density",     __LINE__, __FILE__);
    if (fscanf(fp, "%f\n", &params->accel)        != 1) die("bad accel",       __LINE__, __FILE__);
    if (fscanf(fp, "%f\n", &params->omega)        != 1) die("bad omega",       __LINE__, __FILE__);
    fclose(fp);
  }

  MPI_Bcast(params, sizeof(t_param), MPI_BYTE, 0, MPI_COMM_WORLD);

  /* ---- decomposition ---- */
  compute_decomposition(rank, nprocs, params->ny, decomp);

  const int nx         = params->nx;
  const int local_ny   = decomp->local_ny;
  const int alloc_size = decomp->alloc_rows * nx;

  /* ---- allocate local speed arrays (real + halo rows) ---- */
  t_speed *cells     = malloc(sizeof(t_speed));
  t_speed *tmp_cells = malloc(sizeof(t_speed));
  if (!cells || !tmp_cells)
    die("cannot allocate t_speed structs", __LINE__, __FILE__);

  for (int k = 0; k < NSPEEDS; k++) {
    cells->speeds[k]     = calloc(alloc_size, sizeof(float));
    tmp_cells->speeds[k] = calloc(alloc_size, sizeof(float));
    if (!cells->speeds[k] || !tmp_cells->speeds[k])
      die("cannot allocate speed arrays", __LINE__, __FILE__);
  }

  /* ---- local obstacle array (with halo space, zeroed) ---- */
  int *local_obs = calloc(alloc_size, sizeof(int));
  if (!local_obs) die("cannot allocate obstacles", __LINE__, __FILE__);

  /* ---- equilibrium densities (uniform; every rank identical) ---- */
  const float d0 = params->density * 4.f / 9.f;
  const float d1 = params->density       / 9.f;
  const float d2 = params->density       / 36.f;

  for (int jj = 1; jj <= local_ny; jj++) {
    for (int ii = 0; ii < nx; ii++) {
      const int idx = jj * nx + ii;
      cells->speeds[0][idx] = d0;
      cells->speeds[1][idx] = d1;  cells->speeds[2][idx] = d1;
      cells->speeds[3][idx] = d1;  cells->speeds[4][idx] = d1;
      cells->speeds[5][idx] = d2;  cells->speeds[6][idx] = d2;
      cells->speeds[7][idx] = d2;  cells->speeds[8][idx] = d2;
    }
  }

  /* ---- obstacles: rank 0 reads, then scatter slices ---- */
  int *global_obs = NULL;
  if (rank == 0) {
    global_obs = calloc(nx * params->ny, sizeof(int));
    if (!global_obs) die("cannot allocate global obstacles", __LINE__, __FILE__);

    FILE *fp = fopen(obstaclefile, "r");
    if (!fp) die("could not open obstacle file", __LINE__, __FILE__);

    int xx, yy, blocked, ret;
    while ((ret = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF) {
      if (ret != 3)                       die("bad obstacle line",     __LINE__, __FILE__);
      if (xx < 0 || xx > nx - 1)          die("obstacle x out of range", __LINE__, __FILE__);
      if (yy < 0 || yy > params->ny - 1)  die("obstacle y out of range", __LINE__, __FILE__);
      if (blocked != 1)                    die("blocked must be 1",    __LINE__, __FILE__);
      global_obs[xx + yy * nx] = blocked;
    }
    fclose(fp);
  }

  /* build scatter counts / displacements */
  int *scounts = malloc(nprocs * sizeof(int));
  int *sdispls = malloc(nprocs * sizeof(int));
  for (int r = 0; r < nprocs; r++) {
    int b = params->ny / nprocs;
    int m = params->ny % nprocs;
    int rny = b + (r < m ? 1 : 0);
    int rst = r * b + (r < m ? r : m);
    scounts[r] = rny * nx;
    sdispls[r] = rst * nx;
  }

  MPI_Scatterv(global_obs, scounts, sdispls, MPI_INT,
               &local_obs[1 * nx], local_ny * nx, MPI_INT,
               0, MPI_COMM_WORLD);

  free(scounts);
  free(sdispls);
  if (rank == 0) free(global_obs);

  /* ---- av_vels record ---- */
  *av_vels_ptr   = malloc(params->maxIters * sizeof(float));
  if (!*av_vels_ptr) die("cannot allocate av_vels", __LINE__, __FILE__);

  *cells_ptr     = cells;
  *tmp_cells_ptr = tmp_cells;
  *obstacles_ptr = local_obs;
}

/* ================================================================== */
/*  Gather + write output (rank 0 only)                                */
/* ================================================================== */

void gather_and_write(const t_param params, const t_decomp *d,
                      const t_speed *cells, const int *obstacles,
                      const float *av_vels)
{
  const int nx     = params.nx;
  const int ny     = params.ny;
  const int nprocs = d->nprocs;
  const int rank   = d->rank;

  int *rcounts = malloc(nprocs * sizeof(int));
  int *rdispls = malloc(nprocs * sizeof(int));
  for (int r = 0; r < nprocs; r++) {
    int b = ny / nprocs, m = ny % nprocs;
    int rny = b + (r < m ? 1 : 0);
    int rst = r * b + (r < m ? r : m);
    rcounts[r] = rny * nx;
    rdispls[r] = rst * nx;
  }

  float *gspd[NSPEEDS];
  int   *gobs = NULL;
  for (int k = 0; k < NSPEEDS; k++)
    gspd[k] = (rank == 0) ? malloc(nx * ny * sizeof(float)) : NULL;
  if (rank == 0) {
    gobs = malloc(nx * ny * sizeof(int));
    if (!gobs) die("gather: cannot allocate obstacles", __LINE__, __FILE__);
  }

  for (int k = 0; k < NSPEEDS; k++)
    MPI_Gatherv(&cells->speeds[k][1 * nx], d->local_ny * nx, MPI_FLOAT,
                gspd[k], rcounts, rdispls, MPI_FLOAT, 0, MPI_COMM_WORLD);

  MPI_Gatherv(&obstacles[1 * nx], d->local_ny * nx, MPI_INT,
              gobs, rcounts, rdispls, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    const float c_sq = 1.f / 3.f;
    FILE *fp = fopen(FINALSTATEFILE, "w");
    if (!fp) die("could not open final_state.dat", __LINE__, __FILE__);

    for (int jj = 0; jj < ny; jj++) {
      for (int ii = 0; ii < nx; ii++) {
        const int idx = ii + jj * nx;
        float u_x, u_y, u, pressure;

        if (gobs[idx]) {
          u_x = u_y = u = 0.f;
          pressure = params.density * c_sq;
        } else {
          float ld = 0.f;
          for (int k = 0; k < NSPEEDS; k++) ld += gspd[k][idx];
          u_x = (gspd[1][idx]+gspd[5][idx]+gspd[8][idx]
               -(gspd[3][idx]+gspd[6][idx]+gspd[7][idx])) / ld;
          u_y = (gspd[2][idx]+gspd[5][idx]+gspd[6][idx]
               -(gspd[4][idx]+gspd[7][idx]+gspd[8][idx])) / ld;
          u = sqrtf(u_x*u_x + u_y*u_y);
          pressure = ld * c_sq;
        }

        fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n",
                ii, jj, u_x, u_y, u, pressure, gobs[idx]);
      }
    }
    fclose(fp);

    fp = fopen(AVVELSFILE, "w");
    if (!fp) die("could not open av_vels.dat", __LINE__, __FILE__);
    for (int tt = 0; tt < params.maxIters; tt++)
      fprintf(fp, "%d:\t%.12E\n", tt, av_vels[tt]);
    fclose(fp);

    for (int k = 0; k < NSPEEDS; k++) free(gspd[k]);
    free(gobs);
  }

  free(rcounts);
  free(rdispls);
}

/* ================================================================== */
/*  Reynolds number (distributed reduction)                            */
/* ================================================================== */

float calc_reynolds(const t_param params, const t_decomp *d,
                    const t_speed *cells, const int *obstacles)
{
  const int nx       = params.nx;
  const int local_ny = d->local_ny;

  float loc_u = 0.f;
  int   loc_c = 0;

  for (int jj = 1; jj <= local_ny; jj++) {
    for (int ii = 0; ii < nx; ii++) {
      const int idx = jj * nx + ii;
      if (obstacles[idx]) continue;

      float rho = 0.f;
      for (int k = 0; k < NSPEEDS; k++) rho += cells->speeds[k][idx];
      const float inv = 1.f / rho;

      const float ux = (cells->speeds[1][idx]+cells->speeds[5][idx]+cells->speeds[8][idx]
                       -(cells->speeds[3][idx]+cells->speeds[6][idx]+cells->speeds[7][idx]))*inv;
      const float uy = (cells->speeds[2][idx]+cells->speeds[5][idx]+cells->speeds[6][idx]
                       -(cells->speeds[4][idx]+cells->speeds[7][idx]+cells->speeds[8][idx]))*inv;
      loc_u += sqrtf(ux*ux + uy*uy);
      loc_c++;
    }
  }

  float glb_u;
  int   glb_c;
  MPI_Reduce(&loc_u, &glb_u, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&loc_c, &glb_c, 1, MPI_INT,   MPI_SUM, 0, MPI_COMM_WORLD);

  if (d->rank == 0) {
    const float visc = 1.f/6.f * (2.f/params.omega - 1.f);
    float av = (glb_c > 0) ? glb_u / (float)glb_c : 0.f;
    return av * params.reynolds_dim / visc;
  }
  return 0.f;
}

/* ================================================================== */
/*  Finalise                                                           */
/* ================================================================== */

void finalise(t_speed **cells_ptr, t_speed **tmp_cells_ptr,
              int **obstacles_ptr, float **av_vels_ptr)
{
  for (int k = 0; k < NSPEEDS; k++) {
    free((*cells_ptr)->speeds[k]);
    free((*tmp_cells_ptr)->speeds[k]);
  }
  free(*cells_ptr);     *cells_ptr     = NULL;
  free(*tmp_cells_ptr); *tmp_cells_ptr = NULL;
  free(*obstacles_ptr); *obstacles_ptr = NULL;
  free(*av_vels_ptr);   *av_vels_ptr   = NULL;
}

/* ================================================================== */
/*  Utilities                                                          */
/* ================================================================== */

void die(const char *msg, const int line, const char *file)
{
  fprintf(stderr, "Error at line %d of file %s:\n%s\n", line, file, msg);
  fflush(stderr);
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}