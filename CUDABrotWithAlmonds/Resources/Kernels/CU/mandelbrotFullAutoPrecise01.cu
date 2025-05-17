__device__ bool isInteresting(double x0, double y0, int maxIter) {
    double x = 0.0;
    double y = 0.0;
    int iter = 0;
    while (x*x + y*y <= 4.0 && iter < maxIter) {
        double xtemp = x*x - y*y + x0;
        y = 2.0*x*y + y0;
        x = xtemp;
        iter++;
    }
    return iter > maxIter / 4 && iter < maxIter - 10;
}

__device__ void findInterestingOffset(double* offsetX, double* offsetY, double zoom, int maxIter) {
    const double rangeX = 3.0; // [-2.0, 1.0]
    const double rangeY = 3.0; // [-1.5, 1.5]
    const int samples = 5;
    for (int sx = 0; sx < samples; ++sx) {
        for (int sy = 0; sy < samples; ++sy) {
            double x0 = -2.0 + (rangeX * sx / samples);
            double y0 = -1.5 + (rangeY * sy / samples);
            if (isInteresting(x0, y0, maxIter)) {
                *offsetX = x0;
                *offsetY = y0;
                return;
            }
        }
    }
    // Fallback
    *offsetX = -0.7436438870371587;
    *offsetY =  0.13182590420531197;
}

extern "C" __global__ void mandelbrotFullAutoPrecise01(
    const unsigned char* inputPixels,
    unsigned char* outputPixels,
    int width,
    int height,
    double zoom,
    int iterCoeff,
    int baseR,
    int baseG,
    int baseB)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    iterCoeff = max(1, min(iterCoeff, 1000));
    int maxIter = 100 + static_cast<int>(iterCoeff * log(zoom + 1.0));

    __shared__ double offsetX;
    __shared__ double offsetY;

    // Einmal pro Block: Offset berechnen
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        findInterestingOffset(&offsetX, &offsetY, zoom, maxIter);
    }
    __syncthreads();

    if (px >= width || py >= height) return;

    double x0 = (px - width / 2.0) / (width / 2.0) / zoom + offsetX;
    double y0 = (py - height / 2.0) / (height / 2.0) / zoom + offsetY;

    double x = 0.0;
    double y = 0.0;
    int iter = 0;

    while (x * x + y * y <= 4.0 && iter < maxIter) {
        double xtemp = x * x - y * y + x0;
        y = 2.0 * x * y + y0;
        x = xtemp;
        iter++;
    }

    int idx = (py * width + px) * 4;

    if (iter == maxIter) {
        outputPixels[idx + 0] = baseR;
        outputPixels[idx + 1] = baseG;
        outputPixels[idx + 2] = baseB;
    } else {
        float t = (float)iter / (float)maxIter;
        float r = __sinf(t * 3.14159f) * 255.0f;
        float g = __sinf(t * 6.28318f + 1.0472f) * 255.0f;
        float b = __sinf(t * 9.42477f + 2.0944f) * 255.0f;

        outputPixels[idx + 0] = min(255, baseR + (int)(r * (1.0f - t)));
        outputPixels[idx + 1] = min(255, baseG + (int)(g * (1.0f - t)));
        outputPixels[idx + 2] = min(255, baseB + (int)(b * (1.0f - t)));
    }

    outputPixels[idx + 3] = 255;
}
