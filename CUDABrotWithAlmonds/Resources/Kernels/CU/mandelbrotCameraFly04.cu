extern "C" __global__ void mandelbrotCameraFly04(
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

    // 🌀 Kamerafahrt abhängig vom log(zoom)
    double t = log(zoom + 1.0); // logarithmisch stabil bei großen Zooms

    // Sinus-basierter Flug durch Mandelbrotwelt
    double offsetX = -0.75 + 0.2 * sin(t * 0.2) + 0.05 * cos(t * 1.5);
    double offsetY =  0.0  + 0.15 * cos(t * 0.3) + 0.05 * sin(t * 2.5);

    // 🖼️ Normalisiertes Bild auf komplexe Ebene
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

    // 🎨 Farbgebung
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
