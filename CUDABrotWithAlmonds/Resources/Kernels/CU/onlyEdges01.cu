// Hilfsfunktionen außerhalb des Kernels definieren
__device__ float3 scale3(float3 v, float s) {
    return make_float3(v.x * s, v.y * s, v.z * s);
}

__device__ float3 add3(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

extern "C" __global__ void onlyEdges01(
    const unsigned char* inputPixels,
    unsigned char* outputPixels,
    int width,
    int height,
    float threshold,
    int thickness,
    int edgeR,
    int edgeG,
    int edgeB)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int pixelPos = (y * width + x) * 4;

    // R/B-Tausch vorbereiten
    unsigned char clampedB = (unsigned char)min(max(edgeR, 0), 255);  // R -> B
    unsigned char clampedG = (unsigned char)min(max(edgeG, 0), 255);
    unsigned char clampedR = (unsigned char)min(max(edgeB, 0), 255);  // B -> R
    int clampedThickness = max(0, min(thickness, 10));
    float absThreshold = fabsf(threshold);

    // Weißer Hintergrund
    outputPixels[pixelPos + 0] = 255;
    outputPixels[pixelPos + 1] = 255;
    outputPixels[pixelPos + 2] = 255;
    outputPixels[pixelPos + 3] = 255;

    if (x >= clampedThickness && x < width - clampedThickness &&
        y >= clampedThickness && y < height - clampedThickness)
    {
        const int sobelX[3][3] = {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1}
        };
        const int sobelY[3][3] = {
            {-1, -2, -1},
            { 0,  0,  0},
            { 1,  2,  1}
        };

        float3 gradX = make_float3(0.0f, 0.0f, 0.0f);
        float3 gradY = make_float3(0.0f, 0.0f, 0.0f);

        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                int neighborPos = (ny * width + nx) * 4;

                float3 rgb = make_float3(
                    inputPixels[neighborPos + 2] / 255.0f, // B -> R
                    inputPixels[neighborPos + 1] / 255.0f, // G
                    inputPixels[neighborPos + 0] / 255.0f  // R -> B
                );

                int kx = sobelX[dy + 1][dx + 1];
                int ky = sobelY[dy + 1][dx + 1];

                gradX = add3(gradX, scale3(rgb, (float)kx));
                gradY = add3(gradY, scale3(rgb, (float)ky));
            }
        }

        float3 magnitude = make_float3(
            sqrtf(gradX.x * gradX.x + gradY.x * gradY.x),
            sqrtf(gradX.y * gradX.y + gradY.y * gradY.y),
            sqrtf(gradX.z * gradX.z + gradY.z * gradY.z)
        );

        float avgMagnitude = (magnitude.x + magnitude.y + magnitude.z) / 3.0f;

        if (avgMagnitude > absThreshold) {
            for (int dy = -clampedThickness; dy <= clampedThickness; dy++) {
                for (int dx = -clampedThickness; dx <= clampedThickness; dx++) {
                    if (dx*dx + dy*dy <= clampedThickness*clampedThickness) {
                        int px = x + dx;
                        int py = y + dy;
                        if (px >= 0 && px < width && py >= 0 && py < height) {
                            int writePos = (py * width + px) * 4;
                            outputPixels[writePos + 0] = clampedR;
                            outputPixels[writePos + 1] = clampedG;
                            outputPixels[writePos + 2] = clampedB;
                            outputPixels[writePos + 3] = 255;
                        }
                    }
                }
            }
        }
    }
}
