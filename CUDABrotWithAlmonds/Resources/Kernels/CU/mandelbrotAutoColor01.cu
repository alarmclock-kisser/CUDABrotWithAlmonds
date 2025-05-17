extern "C" __global__ void mandelbrotAutoColor01(
    const unsigned char* inputPixels,
    unsigned char* outputPixels,
    int width,
    int height,
    float zoom,
    float offsetX,
    float offsetY,
    int iterCoeff,
    int baseR,
    int baseG,
    int baseB)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    // Clamp iterCoeff to 1-1000
    iterCoeff = max(1, min(iterCoeff, 1000));

    if (px >= width || py >= height) return;

    // Automatische Iterationsberechnung basierend auf Zoom
    int maxIter = 100 + static_cast<int>(iterCoeff * __logf(zoom + 1.0f));

    // Koordinatenberechnung
    float x0 = (px - width/2.0f) / (width/2.0f) / zoom + offsetX;
    float y0 = (py - height/2.0f) / (height/2.0f) / zoom + offsetY;

    float x = 0.0f;
    float y = 0.0f;
    int iter = 0;

    // Mandelbrot-Iteration
    while (x*x + y*y <= 4.0f && iter < maxIter)
    {
        float xtemp = x*x - y*y + x0;
        y = 2.0f*x*y + y0;
        x = xtemp;
        iter++;
    }

    // Farbberechnung mit Basis-Farbverschiebung
    int idx = (py * width + px) * 4;
    
    if (iter == maxIter)
    {
        // Inneres des Fraktals in Basis-Farbe
        outputPixels[idx+0] = baseR;
        outputPixels[idx+1] = baseG;
        outputPixels[idx+2] = baseB;
        outputPixels[idx+3] = 255; // Alpha
    }
    else
    {
        // Farbverlauf basierend auf Iterationen + Basis-Farbverschiebung
        float t = (float)iter / (float)maxIter;
        
        // Nicht-lineare Farbübergänge für besseren Kontrast
        float r = __sinf(t * 3.14159f) * 255.0f;
        float g = __sinf(t * 6.28318f + 1.0472f) * 255.0f; // + 60°
        float b = __sinf(t * 9.42477f + 2.0944f) * 255.0f; // + 120°
        
        // Mit Basis-Farbe mischen
        outputPixels[idx+0] = min(255, baseR + (int)(r * (1.0f - t)));
        outputPixels[idx+1] = min(255, baseG + (int)(g * (1.0f - t)));
        outputPixels[idx+2] = min(255, baseB + (int)(b * (1.0f - t)));
        outputPixels[idx+3] = 255; // Alpha
    }
}