extern "C" __global__ void hueshift01(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    float shiftValue,
    float rotationValue
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    // Koordinaten relativ zum Mittelpunkt
    float cx = width / 2.0f;
    float cy = height / 2.0f;
    float dx = x - cx;
    float dy = y - cy;

    // Rücktransformation (inverse Rotation)
    float cosA = cosf(-rotationValue);
    float sinA = sinf(-rotationValue);
    float srcXf = cx + dx * cosA - dy * sinA;
    float srcYf = cy + dx * sinA + dy * cosA;

    // Rundung auf Integer (nearest neighbor)
    int srcX = (int)(srcXf + 0.5f);
    int srcY = (int)(srcYf + 0.5f);

    // Check bounds
    if (srcX < 0 || srcX >= width || srcY < 0 || srcY >= height)
    {
        // Schwarz setzen, wenn außerhalb
        int dstIdx = (y * width + x) * 4;
        output[dstIdx + 0] = 0;
        output[dstIdx + 1] = 0;
        output[dstIdx + 2] = 0;
        output[dstIdx + 3] = 255;
        return;
    }

    // Input-Pixel lesen
    int srcIdx = (srcY * width + srcX) * 4;
    float r = input[srcIdx + 0] / 255.0f;
    float g = input[srcIdx + 1] / 255.0f;
    float b = input[srcIdx + 2] / 255.0f;
    unsigned char a = input[srcIdx + 3];

    // RGB -> HSV
    float maxC = fmaxf(r, fmaxf(g, b));
    float minC = fminf(r, fminf(g, b));
    float delta = maxC - minC;

    float h = 0.0f;
    if (delta != 0.0f)
    {
        if (maxC == r) h = fmodf(((g - b) / delta), 6.0f);
        else if (maxC == g) h = ((b - r) / delta) + 2.0f;
        else h = ((r - g) / delta) + 4.0f;

        h /= 6.0f;
        if (h < 0.0f) h += 1.0f;
    }

    float s = maxC == 0.0f ? 0.0f : delta / maxC;
    float v = maxC;

    // Hue verschieben
    h += shiftValue;
    if (h > 1.0f) h -= 1.0f;
    if (h < 0.0f) h += 1.0f;

    // HSV -> RGB
    float c = v * s;
    float hh = h * 6.0f;
    float x1 = c * (1 - fabsf(fmodf(hh, 2.0f) - 1.0f));
    float m = v - c;

    float r1, g1, b1;
    if (hh < 1)      { r1 = c;  g1 = x1; b1 = 0; }
    else if (hh < 2) { r1 = x1; g1 = c;  b1 = 0; }
    else if (hh < 3) { r1 = 0;  g1 = c;  b1 = x1; }
    else if (hh < 4) { r1 = 0;  g1 = x1; b1 = c; }
    else if (hh < 5) { r1 = x1; g1 = 0;  b1 = c; }
    else             { r1 = c;  g1 = 0;  b1 = x1; }

    int dstIdx = (y * width + x) * 4;
    output[dstIdx + 0] = (unsigned char)(fminf(fmaxf((r1 + m) * 255.0f, 0.0f), 255.0f));
    output[dstIdx + 1] = (unsigned char)(fminf(fmaxf((g1 + m) * 255.0f, 0.0f), 255.0f));
    output[dstIdx + 2] = (unsigned char)(fminf(fmaxf((b1 + m) * 255.0f, 0.0f), 255.0f));
    output[dstIdx + 3] = a;
}
