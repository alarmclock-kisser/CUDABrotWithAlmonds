extern "C" __global__ void cubeRenderer02(
    unsigned char* output,
    int width,
    int height,
    float zoom,
    float rotationX,
    float rotationY,
    float rotationZ,
    int thickness,
    int edgeR,
    int edgeG,
    int edgeB,
    int backR,
    int backG,
    int backB
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;

    // Setze Hintergrundfarbe
    output[idx + 0] = backR;
    output[idx + 1] = backG;
    output[idx + 2] = backB;

    // Würfelpunkte im Raum (-1 bis +1)
    float3 cubeVerts[8] = {
        {-1, -1, -1},
        { 1, -1, -1},
        { 1,  1, -1},
        {-1,  1, -1},
        {-1, -1,  1},
        { 1, -1,  1},
        { 1,  1,  1},
        {-1,  1,  1}
    };

    // Kantenpaare (12 Linien)
    int edges[12][2] = {
        {0,1},{1,2},{2,3},{3,0},
        {4,5},{5,6},{6,7},{7,4},
        {0,4},{1,5},{2,6},{3,7}
    };

    // Rotation (Z * Y * X)
    float cx = cosf(rotationX), sx = sinf(rotationX);
    float cy = cosf(rotationY), sy = sinf(rotationY);
    float cz = cosf(rotationZ), sz = sinf(rotationZ);

    // Projizierte 2D-Koordinaten
    float2 projected[8];
    for (int i = 0; i < 8; ++i) {
        float3 p = cubeVerts[i];

        // Rotate X
        float y1 = cx * p.y - sx * p.z;
        float z1 = sx * p.y + cx * p.z;
        p.y = y1; p.z = z1;

        // Rotate Y
        float x2 = cy * p.x + sy * p.z;
        float z2 = -sy * p.x + cy * p.z;
        p.x = x2; p.z = z2;

        // Rotate Z
        float x3 = cz * p.x - sz * p.y;
        float y3 = sz * p.x + cz * p.y;
        p.x = x3; p.y = y3;

        // Perspektivische Projektion
        float scale = zoom / (p.z + 4.0f); // +4 stabilisiert Z
        int px = (int)(p.x * scale * width * 0.4f + width * 0.5f);
        int py = (int)(p.y * scale * height * 0.4f + height * 0.5f);

        projected[i] = make_float2(px, py);
    }

    // Linien zeichnen (über Kanten)
    for (int i = 0; i < 12; ++i) {
        float2 p1 = projected[edges[i][0]];
        float2 p2 = projected[edges[i][1]];

        float2 dir = make_float2(p2.x - p1.x, p2.y - p1.y);
        float len2 = dir.x * dir.x + dir.y * dir.y;
        if (len2 < 1e-5f) continue;

        float2 px = make_float2(x, y);
        float2 ap = make_float2(px.x - p1.x, px.y - p1.y);
        float t = fmaxf(0.0f, fminf(1.0f, (ap.x * dir.x + ap.y * dir.y) / len2));
        float2 closest = make_float2(p1.x + t * dir.x, p1.y + t * dir.y);

        float dx = closest.x - px.x;
        float dy = closest.y - px.y;
        float dist2 = dx * dx + dy * dy;

        if (dist2 < thickness * thickness) {
            output[idx + 0] = edgeR;
            output[idx + 1] = edgeG;
            output[idx + 2] = edgeB;
        }
    }
}