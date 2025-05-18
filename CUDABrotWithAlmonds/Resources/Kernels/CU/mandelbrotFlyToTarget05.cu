extern "C" __global__ void mandelbrotFlyToTarget05(
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
    if (px >= width || py >= height) return;

    iterCoeff = max(1, min(iterCoeff, 1000));
    int maxIter = 100 + (int)(log2(zoom + 1.0) * iterCoeff);

    // 🎯 Zielkoordinaten
    double targetX = -0.743643887037151;
    double targetY =  0.13182590420533;

    // 🌀 Kameraflug per logarithmischem Spiralansatz
    double angle = 0.2 * log(zoom + 1.0);
    double radius = 0.0008 / (zoom + 1.0); // kleiner Radius bei großem Zoom

    double offsetX = targetX + radius * cos(angle);
    double offsetY = targetY + radius * sin(angle);

    // Normale Mandelbrot-Projektion
    double x0 = (px - width / 2.0) / (width / 2.0) / zoom + offsetX;
    double y0 = (py - height / 2.0) / (height / 2.0) / zoom + offsetY;

    double x = 0.0;
    double y = 0.0;
    int iter = 0;

    while (x*x + y*y <= 4.0 && iter < maxIter) {
        double xtemp = x*x - y*y + x0;
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
        float tColor = (float)iter / (float)maxIter;
        float r = __sinf(tColor * 3.14159f) * 255.0f;
        float g = __sinf(tColor * 6.28318f + 1.0472f) * 255.0f;
        float b = __sinf(tColor * 9.42477f + 2.0944f) * 255.0f;

        outputPixels[idx + 0] = min(255, baseR + (int)(r * (1.0f - tColor)));
        outputPixels[idx + 1] = min(255, baseG + (int)(g * (1.0f - tColor)));
        outputPixels[idx + 2] = min(255, baseB + (int)(b * (1.0f - tColor)));
    }

    outputPixels[idx + 3] = 255;
}
