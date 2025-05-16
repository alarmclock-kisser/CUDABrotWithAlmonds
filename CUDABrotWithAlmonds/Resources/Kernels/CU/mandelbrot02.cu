extern "C" __global__ void mandelbrot02(
    const unsigned char* inputPixels,
    unsigned char* outputPixels,
    int width,
    int height,
    float zoom,
    float offsetX,
    float offsetY,
    int maxIter)
{
    // Koordinaten pro Thread
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height)
        return;

    // Zentrierte, normalisierte Koordinaten mit Zoom + Offset
    float x0 = (px - width / 2.0f) / (width / 2.0f) / zoom + offsetX;
    float y0 = (py - height / 2.0f) / (height / 2.0f) / zoom + offsetY;

    float x = 0.0f;
    float y = 0.0f;
    int iter = 0;

    // Iteration zur Bestimmung der Divergenz
    while (x * x + y * y <= 4.0f && iter < maxIter)
    {
        float xtemp = x * x - y * y + x0;
        y = 2.0f * x * y + y0;
        x = xtemp;
        iter++;
    }

    int pixelIndex = (py * width + px) * 4;

    // Schwarz für Punkte im Fraktal
    if (iter == maxIter)
    {
        outputPixels[pixelIndex + 0] = 0;     // R
        outputPixels[pixelIndex + 1] = 0;     // G
        outputPixels[pixelIndex + 2] = 0;     // B
        outputPixels[pixelIndex + 3] = 255;   // A
    }
    else
    {
        // Farbschema ähnlich wie Mandelbrot02
        unsigned char r, g, b;

        if (iter < maxIter / 4) {
            r = 255 * iter / (maxIter / 4);
            g = 0;
            b = 0;
        } else if (iter < maxIter / 2) {
            r = 255;
            g = 255 * (iter - maxIter / 4) / (maxIter / 4);
            b = 0;
        } else if (iter < maxIter * 3 / 4) {
            r = 255 - 255 * (iter - maxIter / 2) / (maxIter / 4);
            g = 255;
            b = 0;
        } else {
            r = 0;
            g = 255;
            b = 255 * (iter - maxIter * 3 / 4) / (maxIter / 4);
        }

        outputPixels[pixelIndex + 0] = r;
        outputPixels[pixelIndex + 1] = g;
        outputPixels[pixelIndex + 2] = b;
        outputPixels[pixelIndex + 3] = 255;
    }
}
