#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#ifdef _OPENMP
  #include <omp.h>
#endif

// ----------------------------- 2D SNIC -----------------------------

#define IDX(x, y, width) ((y) * (width) + (x))

struct Centroid {
    float intensity_sum;
    int x_sum, y_sum;
    int count;
};

struct PriorityQueueSNIC {
    int size;
    int capacity;
    float (*PQ)[4];  // dist, x, y, label
};

static void init_priority_queue(PriorityQueueSNIC* pq, int max_size) {
    pq->size = 0;
    pq->capacity = std::max(1, max_size);
    pq->PQ = (float(*)[4])std::malloc(sizeof(float) * 4 * pq->capacity);
    if (!pq->PQ) {
        std::fprintf(stderr, "Failed to allocate priority queue (2D)\n");
        std::exit(1);
    }
}

static inline void insert_min_heap(PriorityQueueSNIC* pq, float dist, int x, int y, int label) {
    if (pq->size >= pq->capacity) return; // saturate
    pq->PQ[pq->size][0] = dist;
    pq->PQ[pq->size][1] = (float)x;
    pq->PQ[pq->size][2] = (float)y;
    pq->PQ[pq->size][3] = (float)label;
    int idx = pq->size++;

    // up-heap
    while (idx > 0 && pq->PQ[(idx - 1) / 2][0] > pq->PQ[idx][0]) {
        for (int i = 0; i < 4; i++) {
            float tmp = pq->PQ[idx][i];
            pq->PQ[idx][i] = pq->PQ[(idx - 1) / 2][i];
            pq->PQ[(idx - 1) / 2][i] = tmp;
        }
        idx = (idx - 1) / 2;
    }
}

static inline void extract_min(PriorityQueueSNIC* pq, float* dist, int* x, int* y, int* label) {
    if (pq->size == 0) return;

    *dist  = pq->PQ[0][0];
    *x     = (int)pq->PQ[0][1];
    *y     = (int)pq->PQ[0][2];
    *label = (int)pq->PQ[0][3];

    for (int i = 0; i < 4; i++)
        pq->PQ[0][i] = pq->PQ[pq->size - 1][i];

    pq->size--;

    // down-heap
    int idx = 0;
    while (true) {
        int smallest = idx;
        int left = 2 * idx + 1, right = 2 * idx + 2;
        if (left  < pq->size && pq->PQ[left ][0] < pq->PQ[smallest][0]) smallest = left;
        if (right < pq->size && pq->PQ[right][0] < pq->PQ[smallest][0]) smallest = right;
        if (smallest == idx) break;

        for (int i = 0; i < 4; i++) {
            float tmp = pq->PQ[idx][i];
            pq->PQ[idx][i] = pq->PQ[smallest][i];
            pq->PQ[smallest][i] = tmp;
        }
        idx = smallest;
    }
}

static inline float compute_distance(float intensity, int x, int y, const Centroid* c, float S, float m) {
    float cx = (float)c->x_sum / c->count;
    float cy = (float)c->y_sum / c->count;
    float cint = c->intensity_sum / c->count;

    float dx = x - cx;
    float dy = y - cy;
    float dc = intensity - cint;

    return std::sqrt(dc * dc + (m * m / (S * S)) * (dx * dx + dy * dy));
}

void snic_grayscale_heap(const float* image, int width, int height,
                         float spacing, int* labels, float m) {
    const int N = width * height;

    int num_rows = std::max(1, (int)std::floor(height / spacing));
    int num_cols = std::max(1, (int)std::floor(width  / spacing));
    float step_y = (float)height / num_rows;
    float step_x = (float)width  / num_cols;
    int K = num_rows * num_cols;

    // Derived spacing for spatial limit
    float ss = std::sqrt((float)N / (float)K);

    Centroid* centroids = (Centroid*)std::malloc(K * sizeof(Centroid));
    if (!centroids) {
        std::fprintf(stderr, "Failed to allocate centroids (2D)\n");
        return;
    }

    PriorityQueueSNIC pq;
    init_priority_queue(&pq, N);

    for (int i = 0; i < N; i++) labels[i] = -1;

    // Seed placement (guard k < K)
    int k = 0;
    for (int i = 0; i < num_rows; ++i) {
        int y = (int)(step_y * (i + 0.5f));
        if (y >= height) continue;
        for (int j = 0; j < num_cols; ++j) {
            if (k >= K) break;
            int x = (int)(step_x * (j + 0.5f));
            if (x >= width) continue;

            int idx = IDX(x, y, width);
            centroids[k] = (Centroid){ image[idx], x, y, 1 };
            insert_min_heap(&pq, 0.0f, x, y, k);
            k++;
        }
    }

    if (k == 0) {
        std::fprintf(stderr, "No seeds placed! Spacing too large?\n");
        std::free(centroids);
        std::free(pq.PQ);
        return;
    }

    // SNIC propagation (4-connected)
    static const int dx4[4] = { 0, -1,  1,  0};
    static const int dy4[4] = {-1,  0,  0,  1};

    while (pq.size > 0) {
        float dist;
        int x, y, label;
        extract_min(&pq, &dist, &x, &y, &label);

        int idx = IDX(x, y, width);
        if (labels[idx] != -1) continue;

        labels[idx] = label;

        Centroid* c = &centroids[label];
        float intensity = image[idx];

        c->intensity_sum += intensity;
        c->x_sum += x;
        c->y_sum += y;
        c->count += 1;

        float cx = (float)c->x_sum / c->count;
        float cy = (float)c->y_sum / c->count;

        for (int d = 0; d < 4; d++) {
            int nx = x + dx4[d];
            int ny = y + dy4[d];
            if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;

            int nidx = IDX(nx, ny, width);
            if (labels[nidx] != -1) continue;

            float n_intensity = image[nidx];

            float dxs = nx - cx;
            float dys = ny - cy;
            float spatial_sq = dxs * dxs + dys * dys;

            if (spatial_sq <= 4.0f * ss * ss) {
                float dval = compute_distance(n_intensity, nx, ny, c, ss, m);
                insert_min_heap(&pq, dval, nx, ny, label);
            }
        }
    }

    std::free(centroids);
    std::free(pq.PQ);
}

void snic_grayscale_heap_2d_batched(const float* image, int width, int height, int depth,
                                    float spacing, int* labels, float m, int dz) {
    const int num_batches = (depth + dz - 1)/dz;
    int global_label_offset = 0;

    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic)
        for (int b=0;b<num_batches;b++) {
            int z_start = b*dz;
            int z_end = std::min(z_start+dz, depth);
            int local_depth = z_end - z_start;
            int plane = width*height;
            int local_size = plane*local_depth;

            int* local_labels = (int*)std::malloc(sizeof(int)*local_size);
            if (!local_labels) continue;
            for (int i=0;i<local_size;i++) local_labels[i]=-1;

            const float* local_image = &image[(size_t)z_start*plane];

            PriorityQueueSNIC pq;
            init_priority_queue(&pq, local_size);

            // Determine seed grid
            int num_y = std::max(1,(int)std::floor(height/spacing));
            int num_x = std::max(1,(int)std::floor(width/spacing));
            float step_y=(float)height/num_y;
            float step_x=(float)width/num_x;
            float ss = std::sqrt((float)(width*height)/((float)num_x*num_y));

            Centroid* centroids = (Centroid*)std::malloc(sizeof(Centroid)*num_x*num_y*local_depth);
            if (!centroids) { std::free(local_labels); std::free(pq.PQ); continue; }

            int k=0;
            for (int dz_i=0;dz_i<local_depth;dz_i++) {
                const float* slice = &local_image[dz_i*plane];
                for (int i=0;i<num_y;i++) {
                    int y=(int)(step_y*(i+0.5f));
                    if (y>=height) continue;
                    for (int j=0;j<num_x;j++) {
                        int x=(int)(step_x*(j+0.5f));
                        if (x>=width) continue;
                        int idx = IDX(x,y,width);
                        centroids[k]=(Centroid){ slice[idx], x, y, 1 };
                        insert_min_heap(&pq, 0.0f, x, y, k);
                        k++;
                    }
                }
            }

            static const int dx[4]={0,-1,1,0};
            static const int dy[4]={-1,0,0,1};

            while (pq.size>0) {
                float dist; int x,y,label;
                extract_min(&pq,&dist,&x,&y,&label);
                int dz_i = label / (num_x*num_y); // assign slice index based on seed order
                int idx_local = dz_i*plane + y*width + x;
                if (local_labels[idx_local]!=-1) continue;
                local_labels[idx_local]=label;

                Centroid* c = &centroids[label];
                float intensity = local_image[dz_i*plane + y*width + x];
                c->intensity_sum += intensity;
                c->x_sum += x;
                c->y_sum += y;
                c->count += 1;

                float cx=(float)c->x_sum/c->count;
                float cy=(float)c->y_sum/c->count;

                for (int d=0;d<4;d++) {
                    int nx=x+dx[d]; int ny=y+dy[d];
                    if (nx<0||ny<0||nx>=width||ny>=height) continue;
                    int nidx = dz_i*plane + ny*width + nx;
                    if (local_labels[nidx]!=-1) continue;
                    float n_intensity = local_image[nidx];
                    float dxs=nx-cx, dys=ny-cy;
                    float spatial_sq = dxs*dxs + dys*dys;
                    if (spatial_sq <= 4.0f*ss*ss) {
                        float dval = compute_distance(n_intensity,nx,ny,c,ss,m);
                        insert_min_heap(&pq,dval,nx,ny,label);
                    }
                }
            }

            int my_offset=0;
            #pragma omp critical
            {
                my_offset = global_label_offset;
                global_label_offset += k;
            }

            for (int dz_i=0;dz_i<local_depth;dz_i++)
                for (int i=0;i<plane;i++) {
                    int ll = local_labels[dz_i*plane + i];
                    if (ll!=-1) labels[(z_start+dz_i)*plane + i] = ll + my_offset;
                }

            std::free(centroids);
            std::free(local_labels);
            std::free(pq.PQ);
        }
    }
}

// ----------------------------- 3D SNIC -----------------------------

#define IDX3D(x, y, z, width, height) (((z) * (height) + (y)) * (width) + (x))

struct Centroid3d {
    float intensity_sum;
    int x_sum, y_sum, z_sum;
    int count;
};

struct PriorityQueueSNIC3d {
    int size;
    int capacity;
    float (*PQ)[5];  // dist, x, y, z, label
};

static void init_priority_queue_3d_snic(PriorityQueueSNIC3d* pq, int max_size) {
    pq->size = 0;
    pq->capacity = std::max(1, max_size);
    pq->PQ = (float(*)[5])std::malloc(sizeof(float) * 5 * pq->capacity);
    if (!pq->PQ) {
        std::fprintf(stderr, "Failed to allocate priority queue (3D)\n");
        std::exit(1);
    }
}

static inline void insert_min_heap_3d_snic(PriorityQueueSNIC3d* pq, float dist, int x, int y, int z, int label) {
    if (pq->size >= pq->capacity) return; // saturate
    pq->PQ[pq->size][0] = dist;
    pq->PQ[pq->size][1] = (float)x;
    pq->PQ[pq->size][2] = (float)y;
    pq->PQ[pq->size][3] = (float)z;
    pq->PQ[pq->size][4] = (float)label;
    int idx = pq->size++;

    // up-heap
    while (idx > 0 && pq->PQ[(idx - 1) / 2][0] > pq->PQ[idx][0]) {
        for (int i = 0; i < 5; i++) {
            float tmp = pq->PQ[idx][i];
            pq->PQ[idx][i] = pq->PQ[(idx - 1) / 2][i];
            pq->PQ[(idx - 1) / 2][i] = tmp;
        }
        idx = (idx - 1) / 2;
    }
}

static inline void extract_min_3d_snic(PriorityQueueSNIC3d* pq, float* dist, int* x, int* y, int* z, int* label) {
    if (pq->size == 0) return;

    *dist  = pq->PQ[0][0];
    *x     = (int)pq->PQ[0][1];
    *y     = (int)pq->PQ[0][2];
    *z     = (int)pq->PQ[0][3];
    *label = (int)pq->PQ[0][4];

    for (int i = 0; i < 5; i++)
        pq->PQ[0][i] = pq->PQ[pq->size - 1][i];

    pq->size--;

    // down-heap
    int idx = 0;
    while (true) {
        int smallest = idx;
        int left = 2 * idx + 1, right = 2 * idx + 2;
        if (left  < pq->size && pq->PQ[left ][0] < pq->PQ[smallest][0]) smallest = left;
        if (right < pq->size && pq->PQ[right][0] < pq->PQ[smallest][0]) smallest = right;
        if (smallest == idx) break;

        for (int i = 0; i < 5; i++) {
            float tmp = pq->PQ[idx][i];
            pq->PQ[idx][i] = pq->PQ[smallest][i];
            pq->PQ[smallest][i] = tmp;
        }
        idx = smallest;
    }
}

static inline float compute_distance_3d(float intensity, int x1, int y1, int z1,
                                        const Centroid3d* c, float S, float m) {
    float mean_x  = (float)c->x_sum / c->count;
    float mean_y  = (float)c->y_sum / c->count;
    float mean_z  = (float)c->z_sum / c->count;
    float cint    =        c->intensity_sum / c->count;

    float dx = x1 - mean_x;
    float dy = y1 - mean_y;
    float dz = z1 - mean_z;
    float dc = intensity - cint;

    return std::sqrt(dc * dc + (m * m / (S * S)) * (dx * dx + dy * dy + dz * dz));
}

void snic_grayscale_heap_3d(const float* image, int width, int height, int depth,
                            float spacing, int* labels, float m) {
    const int N = width * height * depth;

    int num_z = std::max(1, (int)std::floor(depth  / spacing));
    int num_y = std::max(1, (int)std::floor(height / spacing));
    int num_x = std::max(1, (int)std::floor(width  / spacing));

    float step_z = (float)depth  / num_z;
    float step_y = (float)height / num_y;
    float step_x = (float)width  / num_x;

    const int K_est = num_x * num_y * num_z;
    float ss = std::sqrt((float)N / (float)K_est);

    Centroid3d* centroids = (Centroid3d*)std::malloc(K_est * sizeof(Centroid3d));
    if (!centroids) {
        std::fprintf(stderr, "Failed to allocate centroids (3D)\n");
        return;
    }

    PriorityQueueSNIC3d pq;
    init_priority_queue_3d_snic(&pq, N);

    for (int i = 0; i < N; i++) labels[i] = -1;

    // Seed placement (guard k < K_est)
    int k = 0;
    for (int i = 0; i < num_z; ++i) {
        int z = (int)(step_z * (i + 0.5f));
        if (z >= depth) continue;
        for (int j = 0; j < num_y; ++j) {
            int y = (int)(step_y * (j + 0.5f));
            if (y >= height) continue;
            for (int l = 0; l < num_x; ++l) {
                if (k >= K_est) break;
                int x = (int)(step_x * (l + 0.5f));
                if (x >= width) continue;

                int idx = IDX3D(x, y, z, width, height);
                centroids[k] = (Centroid3d){ image[idx], x, y, z, 1 };
                insert_min_heap_3d_snic(&pq, 0.0f, x, y, z, k);
                k++;
            }
        }
    }

    if (k == 0) {
        std::fprintf(stderr, "No seeds placed! Spacing too large?\n");
        std::free(centroids);
        std::free(pq.PQ);
        return;
    }

    // 6-connected neighbors
    static const int dx6[6] = {-1, 1,  0, 0, 0, 0};
    static const int dy6[6] = { 0, 0, -1, 1, 0, 0};
    static const int dz6[6] = { 0, 0,  0, 0,-1, 1};

    // Region growing
    while (pq.size > 0) {
        float dist;
        int x, y, z, label;
        extract_min_3d_snic(&pq, &dist, &x, &y, &z, &label);

        int idx = IDX3D(x, y, z, width, height);
        if (labels[idx] != -1) continue;

        labels[idx] = label;

        Centroid3d* c = &centroids[label];
        float intensity = image[idx];

        c->intensity_sum += intensity;
        c->x_sum += x;
        c->y_sum += y;
        c->z_sum += z;
        c->count += 1;

        float cx = (float)c->x_sum / c->count;
        float cy = (float)c->y_sum / c->count;
        float cz = (float)c->z_sum / c->count;

        for (int d = 0; d < 6; d++) {
            int nx = x + dx6[d];
            int ny = y + dy6[d];
            int nz = z + dz6[d];
            if (nx < 0 || ny < 0 || nz < 0 || nx >= width || ny >= height || nz >= depth)
                continue;

            int nidx = IDX3D(nx, ny, nz, width, height);
            if (labels[nidx] != -1) continue;

            float n_intensity = image[nidx];

            float dxs = nx - cx;
            float dys = ny - cy;
            float dzs = nz - cz;
            float spatial_sq = dxs * dxs + dys * dys + dzs * dzs;

            if (spatial_sq <= 4.0f * ss * ss) {
                float dval = compute_distance_3d(n_intensity, nx, ny, nz, c, ss, m);
                insert_min_heap_3d_snic(&pq, dval, nx, ny, nz, label);
            }
        }
    }

    std::free(centroids);
    std::free(pq.PQ);
}


// ------------------------- 3D SNIC (Batched in Z) -------------------------

void snic_grayscale_heap_3d_batched(const float* image, int width, int height, int depth,
                                    float spacing, int* labels, float m, int dz) {
    const int num_batches = (depth + dz - 1) / dz;

    // Global running label offset across batches
    int global_label_offset = 0;

    #pragma omp parallel
    {
        // Each thread processes multiple batches (dynamic schedule)
        #pragma omp for schedule(dynamic)
        for (int b = 0; b < num_batches; ++b) {
            const int z_start = b * dz;
            const int z_end   = std::min(z_start + dz, depth);
            const int local_depth = z_end - z_start;
            const int plane = width * height;
            const int local_size = plane * local_depth;

            // Local buffers
            int* local_labels = (int*)std::malloc(sizeof(int) * local_size);
            if (!local_labels) {
                std::fprintf(stderr, "Failed to allocate local_labels (batch %d)\n", b);
                continue;
            }
            for (int i = 0; i < local_size; ++i) local_labels[i] = -1;

            const float* local_image = &image[(size_t)z_start * plane];

            // Local PQ with capacity = local_size
            PriorityQueueSNIC3d pq;
            init_priority_queue_3d_snic(&pq, local_size);

            // Local grid and ss
            int num_z = std::max(1, (int)std::floor(local_depth / spacing));
            int num_y = std::max(1, (int)std::floor(height     / spacing));
            int num_x = std::max(1, (int)std::floor(width      / spacing));

            float step_z = (float)local_depth / num_z;
            float step_y = (float)height      / num_y;
            float step_x = (float)width       / num_x;

            const int K_est = num_x * num_y * num_z;
            float ss = std::sqrt((float)local_size / (float)K_est);

            // Local centroids
            Centroid3d* centroids = (Centroid3d*)std::malloc(sizeof(Centroid3d) * K_est);
            if (!centroids) {
                std::fprintf(stderr, "Failed to allocate centroids (batch %d)\n", b);
                std::free(pq.PQ);
                std::free(local_labels);
                continue;
            }

            // Seed placement
            int k = 0;
            for (int i = 0; i < num_z; ++i) {
                int z = (int)(step_z * (i + 0.5f));
                if (z >= local_depth) continue;
                for (int j = 0; j < num_y; ++j) {
                    int y = (int)(step_y * (j + 0.5f));
                    if (y >= height) continue;
                    for (int l = 0; l < num_x; ++l) {
                        if (k >= K_est) break;
                        int x = (int)(step_x * (l + 0.5f));
                        if (x >= width) continue;

                        int idx_local = IDX3D(x, y, z, width, height); // relative to subvolume base
                        centroids[k] = (Centroid3d){ local_image[idx_local], x, y, z, 1 };
                        insert_min_heap_3d_snic(&pq, 0.0f, x, y, z, k);
                        k++;
                    }
                }
            }

            if (k == 0) {
                std::free(centroids);
                std::free(pq.PQ);
                std::free(local_labels);
                continue;
            }

            // 6-connected neighbors
            static const int dx6[6] = {-1, 1,  0, 0, 0, 0};
            static const int dy6[6] = { 0, 0, -1, 1, 0, 0};
            static const int dz6[6] = { 0, 0,  0, 0,-1, 1};

            // Region growing on subvolume
            while (pq.size > 0) {
                float dist;
                int x, y, z, label;
                extract_min_3d_snic(&pq, &dist, &x, &y, &z, &label);

                int idx_local = IDX3D(x, y, z, width, height);
                if (local_labels[idx_local] != -1) continue;

                local_labels[idx_local] = label;

                Centroid3d* c = &centroids[label];
                float intensity = local_image[idx_local];

                c->intensity_sum += intensity;
                c->x_sum += x;
                c->y_sum += y;
                c->z_sum += z;
                c->count += 1;

                float cx = (float)c->x_sum / c->count;
                float cy = (float)c->y_sum / c->count;
                float cz = (float)c->z_sum / c->count;

                for (int d = 0; d < 6; d++) {
                    int nx = x + dx6[d];
                    int ny = y + dy6[d];
                    int nz = z + dz6[d];
                    if (nx < 0 || ny < 0 || nz < 0 || nx >= width || ny >= height || nz >= local_depth)
                        continue;

                    int nidx_local = IDX3D(nx, ny, nz, width, height);
                    if (local_labels[nidx_local] != -1) continue;

                    float n_intensity = local_image[nidx_local];

                    float dxs = nx - cx;
                    float dys = ny - cy;
                    float dzs = nz - cz;
                    float spatial_sq = dxs * dxs + dys * dys + dzs * dzs;

                    if (spatial_sq <= 4.0f * ss * ss) {
                        float dval = compute_distance_3d(n_intensity, nx, ny, nz, c, ss, m);
                        insert_min_heap_3d_snic(&pq, dval, nx, ny, nz, label);
                    }
                }
            }

            // Reserve a global label offset for this batch (sum of seeds)
            int my_offset = 0;
            #pragma omp critical
            {
                my_offset = global_label_offset;
                global_label_offset += k;
            }

            // Commit local labels into global volume with offset
            for (int z = 0; z < local_depth; ++z) {
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        int local_idx  = IDX3D(x, y, z, width, height);
                        int global_idx = IDX3D(x, y, z + z_start, width, height);
                        int ll = local_labels[local_idx];
                        if (ll != -1) labels[global_idx] = ll + my_offset;
                    }
                }
            }

            std::free(centroids);
            std::free(pq.PQ);
            std::free(local_labels);
        } // for batches
    } // parallel
}
