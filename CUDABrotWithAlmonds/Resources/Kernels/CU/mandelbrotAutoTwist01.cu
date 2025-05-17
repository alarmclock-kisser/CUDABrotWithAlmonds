extern "C" __global__ void mandelbrotAutoTwist01(
    const unsigned char* inputPixels,
    unsigned char* outputPixels,
    int width,
    int height,
    double zoom,
    float offsetX,
    float offsetY,
    char twist,
    int iterCoeff)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    // Clamp iterCoeff
    // max und min sind hier ohne device_launch_parameters.h nicht definiert!
    iterCoeff = max(0, min(iterCoeff, 1000));

    if (px >= width || py >= height) return;

    // Mathematische Konstanten direkt definieren (wie vom Benutzer bereitgestellt)
    #define PI 3.14159265358979323846
    #define TWO_PI 6.28318530717958647692
    #define ONE_THIRD 2.09439510239319549229

    // Dynamische Parameter
    // log ist hier ohne math.h oder math_constants.h nicht definiert!
    double logZoom = log(zoom + 1.0);
    int maxIter = 100 + (int)(iterCoeff * logZoom);

    // Farbrotation (basierend auf logZoom, wie zuvor)
    // fmodf ist hier ohne math.h oder math_constants.h nicht definiert!
    float colorPhase = fmodf((float)(logZoom / 5.0), (float)TWO_PI);

    // Koordinatendrehung basierend auf 'twist' Parameter und logZoom
    // twist (0-255) wird auf einen Faktor (0.0-1.0) skaliert.
    // Bei twist=0 ist der Faktor 0.0, bei twist=255 ist der Faktor 1.0.
    float twistFactor = (float)twist / 255.0f;

    // Die Rotationsgeschwindigkeit wird durch logZoom * twistFactor * Skalierungsfaktor bestimmt.
    // Der Skalierungsfaktor (hier TWO_PI / 10.0f) bestimmt, wie schnell die Drehung
    // mit logZoom zunimmt, wenn twist = 255 ist.
    // Bei twist = 255 (twistFactor = 1.0) ergibt sich eine volle Drehung (2*PI)
    // für jede Zunahme von 10.0 im logZoom.
    // fmodf ist hier ohne math.h oder math_constants.h nicht definiert!
    float coordRotation = fmodf((float)(logZoom * twistFactor * (TWO_PI / 10.0f)), (float)TWO_PI);

    // Cosinus und Sinus für die Drehung
    // cosf und sinf sind hier ohne math.h oder math_constants.h nicht definiert!
    float cosRot = cosf(coordRotation);
    float sinRot = sinf(coordRotation);

    // Ursprungskoordinaten mit hoher Präzision
    double x0 = (px - width/2.0) / (width/2.0) / zoom + offsetX;
    double y0 = (py - height/2.0) / (height/2.0) / zoom + offsetY;

    // Koordinatendrehung anwenden
    double x = x0 * cosRot - y0 * sinRot;
    double y = x0 * sinRot + y0 * cosRot;
    x0 = x; // Die gedrehten Koordinaten werden für die Mandelbrot-Iteration verwendet
    y0 = y;

    // Mandelbrot-Berechnung
    double zx = 0.0;
    double zy = 0.0;
    int iter = 0;

    while (zx*zx + zy*zy <= 4.0 && iter < maxIter)
    {
        double xtemp = zx*zx - zy*zy + x0;
        zy = 2.0*zx*zy + y0;
        zx = xtemp;
        iter++;
    }

    // Farbberechnung
    int idx = (py * width + px) * 4;

    if (iter < maxIter)
    {
        // log2f und fmodf sind hier ohne math.h oder math_constants.h nicht definiert!
        float t = (float)iter / (float)maxIter; // Vereinfachte Farbberechnung wie im ursprünglichen Code
        float r = 0.5f + 0.5f * sinf(colorPhase);
        float g = 0.5f + 0.5f * sinf(colorPhase + ONE_THIRD);
        float b = 0.5f + 0.5f * sinf(colorPhase + 2.0f * ONE_THIRD);

        outputPixels[idx+0] = (unsigned char)(255 * r * t);
        outputPixels[idx+1] = (unsigned char)(255 * g * t);
        outputPixels[idx+2] = (unsigned char)(255 * b * t);
    }
    else
    {
        outputPixels[idx+0] = 0;
        outputPixels[idx+1] = 0;
        outputPixels[idx+2] = 0;
    }
    outputPixels[idx+3] = 255;
}
