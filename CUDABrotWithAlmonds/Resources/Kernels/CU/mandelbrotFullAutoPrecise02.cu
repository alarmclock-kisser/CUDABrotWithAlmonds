extern "C" __global__ void mandelbrotFullAutoPrecise02(
    unsigned char* inputPixels,
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

    // Auto-Iteration abhängig von Zoom
    iterCoeff = max(1, min(iterCoeff, 1000));
    int maxIter = 200 + (int)(iterCoeff * sqrt(log(zoom + 1.0)));

    // === statischer interessanter Punkt mit Zoom-abhängiger Drift ===
    double baseOffsetX = -0.7436438870371587; // Seahorse Valley
    double baseOffsetY =  0.13182590420531197;
    
    // Geringe Drift je Zoom (oszillierend, aber vorhersehbar)
    double offsetX = baseOffsetX + 0.0002 * sin(zoom * 0.1);
    double offsetY = baseOffsetY + 0.0002 * cos(zoom * 0.1);

    // Koordinatentransformation in komplexe Ebene
    double x0 = (px - width / 2.0) / (0.5 * zoom * width) + offsetX;
    double y0 = (py - height / 2.0) / (0.5 * zoom * height) + offsetY;

    // Mandelbrot-Iteration
    double x = 0.0;
    double y = 0.0;
    int iter = 0;

    while (x*x + y*y <= 4.0 && iter < maxIter) {
        double xtemp = x*x - y*y + x0;
        y = 2.0 * x * y + y0;
        x = xtemp;
        iter++;
    }

    // Farbindex berechnen
    int idx = (py * width + px) * 4;

    if (iter >= maxIter) {
        outputPixels[idx + 0] = baseR;
        outputPixels[idx + 1] = baseG;
        outputPixels[idx + 2] = baseB;
    } else {
        // Smooth coloring
        double zn = x * x + y * y;
        double smoothIter = iter + 1.0 - log(log(sqrt(zn))) / log(2.0);
        float t = (float)(smoothIter / maxIter);

        float r = __sinf(3.0f * t * 3.14159f) * 255.0f;
        float g = __sinf(2.0f * t * 3.14159f + 1.0472f) * 255.0f;
        float b = __sinf(1.0f * t * 3.14159f + 2.0944f) * 255.0f;

        outputPixels[idx + 0] = min(255, baseR + (int)(r * (1.0f - t)));
        outputPixels[idx + 1] = min(255, baseG + (int)(g * (1.0f - t)));
        outputPixels[idx + 2] = min(255, baseB + (int)(b * (1.0f - t)));
    }

    outputPixels[idx + 3] = 255;
}

