extern "C" __global__ void kaleidoscope01(
unsigned char* input,
unsigned char* output,
int width,
int height,
int sectors,
float rotation,
float offsetX,
float offsetY)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float cx = width / 2.0f + offsetX * width / 2.0f;
    float cy = height / 2.0f + offsetY * height / 2.0f;

    float dx = x - cx;
    float dy = y - cy;
    float r = sqrtf(dx * dx + dy * dy);
    float angle = atan2f(dy, dx);

    angle -= rotation;

    float sectorAngle = 2.0f * 3.14159265f / sectors;
    angle = fmodf(angle, 2.0f * 3.14159265f);
    if (angle < 0) angle += 2.0f * 3.14159265f;

    int sectorIndex = (int)(angle / sectorAngle);
    float mirroredAngle = angle - sectorIndex * sectorAngle;
    if (sectorIndex % 2 == 1)
        mirroredAngle = sectorAngle - mirroredAngle;

    float finalAngle = mirroredAngle + rotation;

    float srcXf = cx + r * cosf(finalAngle);
    float srcYf = cy + r * sinf(finalAngle);

    int srcX = (int)(srcXf + 0.5f);
    int srcY = (int)(srcYf + 0.5f);

    int dstIdx = (y * width + x) * 4;

    if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height)
    {
        int srcIdx = (srcY * width + srcX) * 4;
        output[dstIdx + 0] = input[srcIdx + 0];
        output[dstIdx + 1] = input[srcIdx + 1];
        output[dstIdx + 2] = input[srcIdx + 2];
        output[dstIdx + 3] = input[srcIdx + 3];
    }
    else
    {
        output[dstIdx + 0] = 0;
        output[dstIdx + 1] = 0;
        output[dstIdx + 2] = 0;
        output[dstIdx + 3] = 255;
    }
}
