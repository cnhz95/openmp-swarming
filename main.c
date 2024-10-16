#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#define v0 1.0	// Velocity
#define eta 0.5	// Random fluctuation in angle (in radians)
#define L 10	// Size of box
#define R 1		// Interaction radius
#define dt 0.2	// Time step
#ifndef Nt
#define Nt 200	// Number of time steps
#endif
#ifndef N
#define N 500	// Number of particles
#endif

void save_to_file(FILE* file, double* x, double* y, double* vx, double* vy, int t) {
	fprintf(file, "Iteration %d\n", t);
	for (int i = 0; i < N; i++) {
		fprintf(file, "%f %f %f %f\n", x[i], y[i], vx[i], vy[i]);
	}
	fprintf(file, "\n\n");
}

double random_range(double min, double max) {
	return min + ((double)random() / RAND_MAX) * (max - min);
}

int main() {
	srandom(time(NULL));
	const int N_THREADS = omp_get_max_threads();
	omp_set_num_threads(N_THREADS);

	FILE *file = fopen("simulation.dat", "w");
	if (!file) {
		perror("Error opening the file!");
		return 1;
	}

	double start_time = omp_get_wtime();

	const double R_SQUARED = R * R;
	double x[N], y[N], theta[N], vx[N], vy[N];
	double mean_theta[N];

	// Initialize bird positions and velocities
	#pragma omp parallel for
	for (int i = 0; i < N; i++) {
		x[i] = random_range(0, L);
		y[i] = random_range(0, L);
		theta[i] = random_range(0, 2 * M_PI);
		vx[i] = v0 * cos(theta[i]);
		vy[i] = v0 * sin(theta[i]);
	}

	// Simulation main loop
	for (int t = 0; t < Nt; t++) {
		#pragma omp parallel
		{
			// Move birds
			#pragma omp for
			for (int i = 0; i < N; i++) {
				x[i] += vx[i] * dt;
				y[i] += vy[i] * dt;

				// Apply periodic boundary conditions
				if (x[i] < 0) x[i] += L;
				if (x[i] >= L) x[i] -= L;
				if (y[i] < 0) y[i] += L;
				if (y[i] >= L) y[i] -= L;
			}

			// Find mean angle of neighbors within R
			double sx[N], sy[N];
			#pragma omp for
			for (int i = 0; i < N; i++) {
				sx[i] = 0.0, sy[i] = 0.0;
				for (int j = 0; j < N; j++) {
					if (pow((x[j] - x[i]), 2) + pow((y[j] - y[i]), 2) < R_SQUARED) {
						sx[i] += cos(theta[j]);
						sy[i] += sin(theta[j]);
					}
				}
				mean_theta[i] = atan2(sy[i], sx[i]);
			}

			// Add random perturbations and update velocities
			#pragma omp for
			for (int i = 0; i < N; i++) {
				theta[i] = mean_theta[i] + eta * (random_range(0, 1) - 0.5);
				vx[i] = v0 * cos(theta[i]);
				vy[i] = v0 * sin(theta[i]);
			}
		}
		#ifdef record
		save_to_file(file, x, y, vx, vy, t);
		#endif
	}

	double end_time = omp_get_wtime();
	printf("Execution time with %d threads: %f seconds\n", N_THREADS, end_time - start_time);
	fclose(file);

	return 0;
}
