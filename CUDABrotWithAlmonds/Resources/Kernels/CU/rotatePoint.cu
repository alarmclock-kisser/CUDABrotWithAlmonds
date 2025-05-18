__device__ void rotatePoint(float& x, float& y, float& z, float rx, float ry, float rz) {
    // Rotation um X-Achse
    float cy = y * cosf(rx) - z * sinf(rx);
    float cz = y * sinf(rx) + z * cosf(rx);
    y = cy;
    z = cz;
    
    // Rotation um Y-Achse
    float cx = x * cosf(ry) - z * sinf(ry);
    cz = x * sinf(ry) + z * cosf(ry);
    x = cx;
    z = cz;
    
    // Rotation um Z-Achse
    cx = x * cosf(rz) - y * sinf(rz);
    cy = x * sinf(rz) + y * cosf(rz);
    x = cx;
    y = cy;
}

__device__ bool isOnEdge(int px, int py, int x1, int y1, int x2, int y2, int thickness) {
    // Vereinfachte Linienalgorithmus mit Dicke
    int minX = min(x1, x2) - thickness;
    int maxX = max(x1, x2) + thickness;
    int minY = min(y1, y2) - thickness;
    int maxY = max(y1, y2) + thickness;
    
    if (px < minX || px > maxX || py < minY || py > maxY) return false;
    
    // Abstand von Punkt zur Linie
    float A = py - y1;
    float B = x1 - px;
    float C = x2 - x1;
    float D = y2 - y1;
    
    float dot = A * C + B * D;
    float len_sq = C * C + D * D;
    float param = (len_sq != 0) ? dot / len_sq : -1;
    
    float xx, yy;
    if (param < 0) {
        xx = x1;
        yy = y1;
    } else if (param > 1) {
        xx = x2;
        yy = y2;
    } else {
        xx = x1 + param * C;
        yy = y1 + param * D;
    }
    
    float dx = px - xx;
    float dy = py - yy;
    float distance = sqrtf(dx * dx + dy * dy);
    
    return distance <= thickness;
}

extern "C" __global__ void cubeRenderer03(
    unsigned char* output,
    int width, int height,
    float zoom,
    float rotationX, float rotationY, float rotationZ,
    int backR, int backG, int backB,
    int edgeR, int edgeG, int edgeB,
    int thickness) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Hintergrundfarbe setzen
    int idx = (y * width + x) * 3;
    output[idx] = backR;
    output[idx+1] = backG;
    output[idx+2] = backB;
    
    // Zentrum des Bildes
    float centerX = width / 2.0f;
    float centerY = height / 2.0f;
    
    // Würfelgröße (4x3 Netz)
    float cubeSize = min(width, height) * 0.2f * zoom;
    
    // 6 Flächen des Würfelnetzes (4x3 Anordnung)
    float faces[6][4][3] = {
        // Obere Reihe (3 Flächen)
        {{-1.5f, 0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f}, {-0.5f, -0.5f, 0.5f}, {-1.5f, -0.5f, 0.5f}}, // links
        {{-0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}, {-0.5f, -0.5f, 0.5f}},   // mitte
        {{0.5f, 0.5f, 0.5f}, {1.5f, 0.5f, 0.5f}, {1.5f, -0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}},     // rechts
        
        // Untere Reihe (3 Flächen)
        {{-0.5f, -0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}, {0.5f, -1.5f, 0.5f}, {-0.5f, -1.5f, 0.5f}}, // unten mitte
        {{-1.5f, -0.5f, -0.5f}, {-0.5f, -0.5f, -0.5f}, {-0.5f, -1.5f, -0.5f}, {-1.5f, -1.5f, -0.5f}}, // unten links
        {{0.5f, -0.5f, -0.5f}, {1.5f, -0.5f, -0.5f}, {1.5f, -1.5f, -0.5f}, {0.5f, -1.5f, -0.5f}}  // unten rechts
    };
    
    // Kanten des Würfelnetzes (jede Fläche hat 4 Kanten)
    int edges[][2] = {
        // Obere Reihe
        {0,1}, {1,2}, {2,3}, {3,0},  // linkes Quadrat
        {4,5}, {5,6}, {6,7}, {7,4},  // mittleres Quadrat
        {8,9}, {9,10}, {10,11}, {11,8}, // rechtes Quadrat
        
        // Untere Reihe
        {12,13}, {13,14}, {14,15}, {15,12},   // unteres mittleres Quadrat
        {16,17}, {17,18}, {18,19}, {19,16}, // unteres linkes Quadrat
        {20,21}, {21,22}, {22,23}, {23,20}, // unteres rechtes Quadrat
        
        // Verbindungskanten zwischen den Quadraten
        {2,6}, {6,10}, {3,7}, {7,11}, {14,18}, {18,22}
    };
    
    // Alle Kanten durchgehen
    for (int e = 0; e < 26; e++) {
        int i1 = edges[e][0];
        int i2 = edges[e][1];
        
        // Flächen- und Punktindex bestimmen
        int face1 = i1 / 4;
        int point1 = i1 % 4;
        int face2 = i2 / 4;
        int point2 = i2 % 4;
        
        // 3D-Punkte holen
        float x1 = faces[face1][point1][0];
        float y1 = faces[face1][point1][1];
        float z1 = faces[face1][point1][2];
        
        float x2 = faces[face2][point2][0];
        float y2 = faces[face2][point2][1];
        float z2 = faces[face2][point2][2];
        
        // Rotation anwenden
        rotatePoint(x1, y1, z1, rotationX, rotationY, rotationZ);
        rotatePoint(x2, y2, z2, rotationX, rotationY, rotationZ);
        
        // Auf Bildschirmkoordinaten projizieren
        int sx1 = (int)(centerX + x1 * cubeSize);
        int sy1 = (int)(centerY + y1 * cubeSize);
        int sx2 = (int)(centerX + x2 * cubeSize);
        int sy2 = (int)(centerY + y2 * cubeSize);
        
        // Prüfen ob Pixel auf der Kante liegt
        if (isOnEdge(x, y, sx1, sy1, sx2, sy2, thickness)) {
            output[idx] = edgeR;
            output[idx+1] = edgeG;
            output[idx+2] = edgeB;
        }
    }
}
