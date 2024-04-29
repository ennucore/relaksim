extern "C"
__global__ void relaxKernel(double *grid, int width, int scale, double epsilon, bool *converged, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int index = idy * width + idx;

    if ((idx < 1) || (idy < 1) || (idx >= width - 1) || (idy >= width - 1) || (index - width < 0) || (index + width >= size) || (idx >= scale && idy >= scale)) {
	//grid[index] = 239.;
        return; // Skip boundary elements
    }

    double old_v = grid[index];
    double new_v = 0.25 * (
        grid[index - 1] +          // left
        grid[index + 1] +          // right
        grid[index - width] +      // top
        grid[index + width]);      // bottom

    if (fabs(new_v - old_v) > epsilon) {
        *converged = false;
    }

    grid[index] = new_v;
}
