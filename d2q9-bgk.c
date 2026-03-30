#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

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
  int    rank;
  int    nprocs;
  int    local_ny;
  int    jj_start;
  int    rank_south;
  int    rank_north;
  int    alloc_rows;
  int    local_fluid_cells;
  float *send_south;
  float *send_north;
  float *recv_south;
  float *recv_north;
} t_decomp;

void compute_decomposition(int rank, int nprocs, int ny, int nx, t_decomp *d);
void free_decomposition(t_decomp *d);

void initialise(const char *paramfile, const char *obstaclefile,
                t_param *params, t_decomp *decomp,
                t_speed **cells_ptr, t_speed **tmp_cells_ptr,
                int **obstacles_ptr, float **av_vels_ptr);

void finalise(t_decomp *decomp, t_speed **cells_ptr, t_speed **tmp_cells_ptr,
              int **obstacles_ptr, float **av_vels_ptr);

void halo_exchange_start(const t_speed *cells, t_decomp *d,
                         int nx, MPI_Request reqs[4]);

void halo_exchange_finish(t_speed *cells, const t_decomp *d,
                          int nx, MPI_Request reqs[4]);

void accelerate_flow(const t_param params, const t_decomp *d,
                     t_speed *cells, const int *obstacles);

void compute_rows(const t_param params, int jj_lo, int jj_hi,
                  const t_speed *cells, t_speed *tmp_cells,
                  const int *obstacles, float *tot_u);

void compute_single_col(const t_param params, int jj_lo, int jj_hi, int ii,
                        int ii_w, int ii_e,
                        const t_speed *cells, t_speed *tmp_cells,
                        const int *obstacles, float *tot_u);

void gather_and_write(const t_param params, const t_decomp *d,
                      const t_speed *cells, const int *obstacles,
                      const float *av_vels);

float calc_reynolds(const t_param params, const t_decomp *d,
                    const t_speed *cells, const int *obstacles);

void die(const char *msg, const int line, const char *file);

static inline void stream_collide_cell(
    float omega, int obs,
    float r0, float r1, float r2, float r3, float r4,
    float r5, float r6, float r7, float r8,
    float *o0, float *o1, float *o2, float *o3, float *o4,
    float *o5, float *o6, float *o7, float *o8,
    float *vel_acc)
{
  const float w0 = 4.f / 9.f;
  const float w1 = 1.f / 9.f;
  const float w2 = 1.f / 36.f;

  float rho = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8;
  if (rho <= 1e-20f) rho = 1e-20f;
  const float inv_rho = 1.f / rho;

  const float ux = (r1 + r5 + r8 - (r3 + r6 + r7)) * inv_rho;
  const float uy = (r2 + r5 + r6 - (r4 + r7 + r8)) * inv_rho;

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

  float col0 = r0 + omega * (feq0 - r0);
  float col1 = r1 + omega * (feq1 - r1);
  float col2 = r2 + omega * (feq2 - r2);
  float col3 = r3 + omega * (feq3 - r3);
  float col4 = r4 + omega * (feq4 - r4);
  float col5 = r5 + omega * (feq5 - r5);
  float col6 = r6 + omega * (feq6 - r6);
  float col7 = r7 + omega * (feq7 - r7);
  float col8 = r8 + omega * (feq8 - r8);

  const float f = (float)(1 - obs);
  const float o = (float)obs;

  *o0 = f * col0 + o * r0;
  *o1 = f * col1 + o * r3;
  *o2 = f * col2 + o * r4;
  *o3 = f * col3 + o * r1;
  *o4 = f * col4 + o * r2;
  *o5 = f * col5 + o * r7;
  *o6 = f * col6 + o * r8;
  *o7 = f * col7 + o * r5;
  *o8 = f * col8 + o * r6;

  *vel_acc += f * sqrtf(u2);
}

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

  int global_fluid_cells;
  MPI_Allreduce(&decomp.local_fluid_cells, &global_fluid_cells, 1,
                MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  const float inv_fluid_cells = (global_fluid_cells > 0)
                              ? 1.f / (float)global_fluid_cells
                              : 0.f;

  gettimeofday(&timstr, NULL);
  double init_toc = timstr.tv_sec + timstr.tv_usec / 1e6;
  double comp_tic = init_toc;

  const int local_ny = decomp.local_ny;

  double t_accel = 0.0, t_halo = 0.0, t_interior = 0.0;
  double t_boundary = 0.0, t_reduce = 0.0;

  //timestep
  for (int tt = 0; tt < params.maxIters; tt++)
  {
    double t0, t1;

    //accelerate
    t0 = MPI_Wtime();
    accelerate_flow(params, &decomp, cells, obstacles);
    t1 = MPI_Wtime();
    t_accel += t1 - t0;

    //halo start
    t0 = MPI_Wtime();
    MPI_Request halo_reqs[4];
    halo_exchange_start(cells, &decomp, params.nx, halo_reqs);
    t1 = MPI_Wtime();
    t_halo += t1 - t0;

    //compute interiors
    t0 = MPI_Wtime();
    float tot_u = 0.f;
    if (local_ny > 2) {
      compute_rows(params, 2, local_ny - 1,
                   cells, tmp_cells, obstacles, &tot_u);
    }
    t1 = MPI_Wtime();
    t_interior += t1 - t0;

    //recv halos and calc boundaries
    t0 = MPI_Wtime();
    halo_exchange_finish(cells, &decomp, params.nx, halo_reqs);
    t1 = MPI_Wtime();
    t_halo += t1 - t0;

    t0 = MPI_Wtime();
    if (local_ny <= 2) {
      compute_rows(params, 1, local_ny,
                   cells, tmp_cells, obstacles, &tot_u);
    } else {
      compute_rows(params, 1, 1,
                   cells, tmp_cells, obstacles, &tot_u);
      compute_rows(params, local_ny, local_ny,
                   cells, tmp_cells, obstacles, &tot_u);
    }
    t1 = MPI_Wtime();
    t_boundary += t1 - t0;

    //reduce
    t0 = MPI_Wtime();
    float global_tot_u;
    MPI_Allreduce(&tot_u, &global_tot_u, 1,
                  MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    av_vels[tt] = global_tot_u * inv_fluid_cells;
    t1 = MPI_Wtime();
    t_reduce += t1 - t0;

    //swap girds
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

  float reynolds = calc_reynolds(params, &decomp, cells, obstacles);

  if (rank == 0) {
    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n", reynolds);
    printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
    printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
    printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
    printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);
    printf("--- Rank 0 compute breakdown ---\n");
    printf("  Accelerate:\t\t%.6lf (s)\n", t_accel);
    printf("  Halo exch:\t\t%.6lf (s)\n",  t_halo);
    printf("  Interior:\t\t%.6lf (s)\n",   t_interior);
    printf("  Boundary:\t\t%.6lf (s)\n",   t_boundary);
    printf("  Allreduce:\t\t%.6lf (s)\n",  t_reduce);
  }

  finalise(&decomp, &cells, &tmp_cells, &obstacles, &av_vels);
  MPI_Finalize();
  return EXIT_SUCCESS;
}


void compute_decomposition(int rank, int nprocs, int ny, int nx, t_decomp *d)
{
  d->rank    = rank;
  d->nprocs  = nprocs;

  int base      = ny / nprocs;
  int remainder  = ny % nprocs;

  d->local_ny = base + (rank < remainder ? 1 : 0);
  d->jj_start = rank * base + (rank < remainder ? rank : remainder);

  d->rank_south = (rank == 0)          ? nprocs - 1 : rank - 1;
  d->rank_north = (rank == nprocs - 1) ? 0          : rank + 1;

  d->alloc_rows = d->local_ny + 2;
  d->local_fluid_cells = 0;

  const int buf_size = NSPEEDS * nx;
  d->send_south = malloc(buf_size * sizeof(float));
  d->send_north = malloc(buf_size * sizeof(float));
  d->recv_south = malloc(buf_size * sizeof(float));
  d->recv_north = malloc(buf_size * sizeof(float));
}

void free_decomposition(t_decomp *d)
{
  free(d->send_south);  d->send_south = NULL;
  free(d->send_north);  d->send_north = NULL;
  free(d->recv_south);  d->recv_south = NULL;
  free(d->recv_north);  d->recv_north = NULL;
}

void halo_exchange_start(const t_speed *cells, t_decomp *d,
                         int nx, MPI_Request reqs[4])
{
  const int local_ny = d->local_ny;
  const int buf_size = NSPEEDS * nx;

  for (int k = 0; k < NSPEEDS; k++)
    memcpy(&d->send_south[k * nx], &cells->speeds[k][1 * nx],
           nx * sizeof(float));

  for (int k = 0; k < NSPEEDS; k++)
    memcpy(&d->send_north[k * nx], &cells->speeds[k][local_ny * nx],
           nx * sizeof(float));

  MPI_Isend(d->send_south, buf_size, MPI_FLOAT,
            d->rank_south, 0, MPI_COMM_WORLD, &reqs[0]);
  MPI_Irecv(d->recv_south, buf_size, MPI_FLOAT,
            d->rank_south, 1, MPI_COMM_WORLD, &reqs[1]);

  MPI_Isend(d->send_north, buf_size, MPI_FLOAT,
            d->rank_north, 1, MPI_COMM_WORLD, &reqs[2]);
  MPI_Irecv(d->recv_north, buf_size, MPI_FLOAT,
            d->rank_north, 0, MPI_COMM_WORLD, &reqs[3]);
}

void halo_exchange_finish(t_speed *cells, const t_decomp *d,
                          int nx, MPI_Request reqs[4])
{
  const int local_ny = d->local_ny;
  MPI_Status stats[4];
  MPI_Waitall(4, reqs, stats);

  for (int k = 0; k < NSPEEDS; k++)
    memcpy(&cells->speeds[k][0],
           &d->recv_south[k * nx], nx * sizeof(float));

  for (int k = 0; k < NSPEEDS; k++)
    memcpy(&cells->speeds[k][(local_ny + 1) * nx],
           &d->recv_north[k * nx], nx * sizeof(float));
}

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

void compute_single_col(const t_param params, int jj_lo, int jj_hi,
                        int ii, int ii_w, int ii_e,
                        const t_speed *cells, t_speed *tmp_cells,
                        const int *obstacles, float *tot_u)
{
  const int   nx    = params.nx;
  const float omega = params.omega;

  for (int jj = jj_lo; jj <= jj_hi; ++jj)
  {
    const int row   = jj * nx;
    const int row_s = (jj - 1) * nx;
    const int row_n = (jj + 1) * nx;
    const int idx   = row + ii;

    float r0 = cells->speeds[0][idx];
    float r1 = cells->speeds[1][row   + ii_w];
    float r2 = cells->speeds[2][row_s + ii];
    float r3 = cells->speeds[3][row   + ii_e];
    float r4 = cells->speeds[4][row_n + ii];
    float r5 = cells->speeds[5][row_s + ii_w];
    float r6 = cells->speeds[6][row_s + ii_e];
    float r7 = cells->speeds[7][row_n + ii_e];
    float r8 = cells->speeds[8][row_n + ii_w];

    float vel = 0.f;
    stream_collide_cell(omega, obstacles[idx],
                        r0,r1,r2,r3,r4,r5,r6,r7,r8,
                        &tmp_cells->speeds[0][idx],
                        &tmp_cells->speeds[1][idx],
                        &tmp_cells->speeds[2][idx],
                        &tmp_cells->speeds[3][idx],
                        &tmp_cells->speeds[4][idx],
                        &tmp_cells->speeds[5][idx],
                        &tmp_cells->speeds[6][idx],
                        &tmp_cells->speeds[7][idx],
                        &tmp_cells->speeds[8][idx],
                        &vel);
    *tot_u += vel;
  }
}

void compute_rows(const t_param params, int jj_lo, int jj_hi,
                  const t_speed *cells, t_speed *tmp_cells,
                  const int *obstacles, float *tot_u)
{
  const int   nx    = params.nx;
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

  float acc_u = 0.f;

  compute_single_col(params, jj_lo, jj_hi, 0, nx - 1, 1,
                     cells, tmp_cells, obstacles, &acc_u);

  if (nx > 1) {
    compute_single_col(params, jj_lo, jj_hi, nx - 1, nx - 2, 0,
                       cells, tmp_cells, obstacles, &acc_u);
  }

  for (int jj = jj_lo; jj <= jj_hi; ++jj)
  {
    const int row   = jj * nx;
    const int row_s = (jj - 1) * nx;
    const int row_n = (jj + 1) * nx;

    for (int ii = 1; ii < nx - 1; ++ii)
    {
      const int idx  = row + ii;

      const float r0 = c0[idx];
      const float r1 = c1[row   + ii - 1];
      const float r2 = c2[row_s + ii];
      const float r3 = c3[row   + ii + 1];
      const float r4 = c4[row_n + ii];
      const float r5 = c5[row_s + ii - 1];
      const float r6 = c6[row_s + ii + 1];
      const float r7 = c7[row_n + ii + 1];
      const float r8 = c8[row_n + ii - 1];

      float vel = 0.f;
      stream_collide_cell(omega, obstacles[idx],
                          r0,r1,r2,r3,r4,r5,r6,r7,r8,
                          &out0[idx],&out1[idx],&out2[idx],
                          &out3[idx],&out4[idx],&out5[idx],
                          &out6[idx],&out7[idx],&out8[idx],
                          &vel);
      acc_u += vel;
    }
  }

  *tot_u += acc_u;
}

void initialise(const char *paramfile, const char *obstaclefile,
                t_param *params, t_decomp *decomp,
                t_speed **cells_ptr, t_speed **tmp_cells_ptr,
                int **obstacles_ptr, float **av_vels_ptr)
{
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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

  compute_decomposition(rank, nprocs, params->ny, params->nx, decomp);

  const int nx         = params->nx;
  const int local_ny   = decomp->local_ny;
  const int alloc_size = decomp->alloc_rows * nx;

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

  int *local_obs = calloc(alloc_size, sizeof(int));
  if (!local_obs) die("cannot allocate obstacles", __LINE__, __FILE__);

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

  int *global_obs = NULL;
  if (rank == 0) {
    global_obs = calloc(nx * params->ny, sizeof(int));
    if (!global_obs) die("cannot allocate global obstacles", __LINE__, __FILE__);

    FILE *fp = fopen(obstaclefile, "r");
    if (!fp) die("could not open obstacle file", __LINE__, __FILE__);

    int xx, yy, blocked, ret;
    while ((ret = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF) {
      if (ret != 3)                        die("bad obstacle line",       __LINE__, __FILE__);
      if (xx < 0 || xx > nx - 1)           die("obstacle x out of range", __LINE__, __FILE__);
      if (yy < 0 || yy > params->ny - 1)   die("obstacle y out of range", __LINE__, __FILE__);
      if (blocked != 1)                     die("blocked must be 1",      __LINE__, __FILE__);
      global_obs[xx + yy * nx] = blocked;
    }
    fclose(fp);
  }

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

  int fluid_count = 0;
  for (int jj = 1; jj <= local_ny; jj++)
    for (int ii = 0; ii < nx; ii++)
      if (!local_obs[jj * nx + ii]) fluid_count++;
  decomp->local_fluid_cells = fluid_count;

  *av_vels_ptr   = malloc(params->maxIters * sizeof(float));
  if (!*av_vels_ptr) die("cannot allocate av_vels", __LINE__, __FILE__);

  *cells_ptr     = cells;
  *tmp_cells_ptr = tmp_cells;
  *obstacles_ptr = local_obs;
}

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

void finalise(t_decomp *decomp, t_speed **cells_ptr, t_speed **tmp_cells_ptr,
              int **obstacles_ptr, float **av_vels_ptr)
{
  free_decomposition(decomp);
  for (int k = 0; k < NSPEEDS; k++) {
    free((*cells_ptr)->speeds[k]);
    free((*tmp_cells_ptr)->speeds[k]);
  }
  free(*cells_ptr);     *cells_ptr     = NULL;
  free(*tmp_cells_ptr); *tmp_cells_ptr = NULL;
  free(*obstacles_ptr); *obstacles_ptr = NULL;
  free(*av_vels_ptr);   *av_vels_ptr   = NULL;
}

void die(const char *msg, const int line, const char *file)
{
  fprintf(stderr, "Error at line %d of file %s:\n%s\n", line, file, msg);
  fflush(stderr);
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}