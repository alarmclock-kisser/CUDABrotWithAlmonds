extern "C" __global__ void downsample01(
    unsigned char* pixels,
    int width,
    int height,
    int bpc)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = (y * width + x) * 4; // 4 channels: R, G, B, A

    // Calculate bitmask for target bit depth
    int shift = 8 - bitsPerChannel;
    unsigned char mask = (0xFF << shift) & 0xFF;

    // Reduce bit depth for RGB channels
    pixels[idx]     &= mask; // R
    pixels[idx + 1] &= mask; // G
    pixels[idx + 2] &= mask; // B
    
    // Optional: Alpha channel
    // pixels[idx + 3] &= mask;
}