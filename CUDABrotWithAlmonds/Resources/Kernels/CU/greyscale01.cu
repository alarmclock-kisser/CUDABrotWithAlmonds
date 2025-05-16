extern "C" __global__ void greyscale01(
    const unsigned char* inputPixels,
    unsigned char* outputPixels,
    int width,
    int height,
    int channels,
    unsigned char intensity) // intensity: 0 = kein Effekt, 255 = voll Graustufen
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int pixelIndex = (y * width + x) * channels;

    // Nur RGB oder RGBA unterstützen
    if (channels < 3)
        return;

    unsigned char r = inputPixels[pixelIndex + 0];
    unsigned char g = inputPixels[pixelIndex + 1];
    unsigned char b = inputPixels[pixelIndex + 2];

    // Standardluminanzformel für Graustufen
    unsigned char gray = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);

    // Stärke mit "intensity" überblenden
    float alpha = intensity / 255.0f;

    unsigned char outR = static_cast<unsigned char>(r * (1.0f - alpha) + gray * alpha);
    unsigned char outG = static_cast<unsigned char>(g * (1.0f - alpha) + gray * alpha);
    unsigned char outB = static_cast<unsigned char>(b * (1.0f - alpha) + gray * alpha);

    outputPixels[pixelIndex + 0] = outR;
    outputPixels[pixelIndex + 1] = outG;
    outputPixels[pixelIndex + 2] = outB;

    // Falls Alpha-Kanal vorhanden, übernehmen
    if (channels == 4)
        outputPixels[pixelIndex + 3] = inputPixels[pixelIndex + 3];
}
