#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>

#define NOMINMAX
#include "tiny_obj_loader.h"
#include "nanort.h"

#define USE_MULTIHIT_RAY_TRAVERSAL (0)

namespace {


struct double3 {
  double3() {}
  double3(double xx, double yy, double zz) {
    x = xx;
    y = yy;
    z = zz;
  }
  double3(const double *p) {
    x = p[0];
    y = p[1];
    z = p[2];
  }

  double3 operator*(double f) const { return double3(x * f, y * f, z * f); }
  double3 operator-(const double3 &f2) const {
    return double3(x - f2.x, y - f2.y, z - f2.z);
  }
  double3 operator-() const { return double3(-x, -y, -z); }
  double3 operator*(const double3 &f2) const {
    return double3(x * f2.x, y * f2.y, z * f2.z);
  }
  double3 operator+(const double3 &f2) const {
    return double3(x + f2.x, y + f2.y, z + f2.z);
  }
  double3 &operator+=(const double3 &f2) {
    x += f2.x;
    y += f2.y;
    z += f2.z;
    return (*this);
  }
  double3 &operator*=(const double3 &f2) {
    x *= f2.x;
    y *= f2.y;
    z *= f2.z;
    return (*this);
  }
  double3 &operator*=(const double &f2) {
    x *= f2;
    y *= f2;
    z *= f2;
    return (*this);
  }
  double3 operator/(const double3 &f2) const {
    return double3(x / f2.x, y / f2.y, z / f2.z);
  }
  double3 operator/(const double &f2) const {
    return double3(x / f2, y / f2, z / f2);
  }
  double operator[](int i) const { return (&x)[i]; }
  double &operator[](int i) { return (&x)[i]; }

  double3 neg() { return double3(-x, -y, -z); }

  double length() { return sqrtf(x * x + y * y + z * z); }

  void normalize() {
    double len = length();
    if (fabs(len) > 1.0e-6) {
      double inv_len = 1.0 / len;
      x *= inv_len;
      y *= inv_len;
      z *= inv_len;
    }
  }

  double x, y, z;
  // double pad;  // for alignment
};

inline double3 normalize(double3 v) {
  v.normalize();
  return v;
}

inline double3 operator*(double f, const double3 &v) {
  return double3(v.x * f, v.y * f, v.z * f);
}

inline double3 vcross(double3 a, double3 b) {
  double3 c;
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
  return c;
}

typedef struct {
  size_t num_vertices;
  size_t num_faces;
  double *vertices;                   /// [xyz] * num_vertices
  double *facevarying_normals;        /// [xyz] * 3(triangle) * num_faces
  double *facevarying_tangents;       /// [xyz] * 3(triangle) * num_faces
  double *facevarying_binormals;      /// [xyz] * 3(triangle) * num_faces
  double *facevarying_uvs;            /// [xyz] * 3(triangle) * num_faces
  double *facevarying_vertex_colors;  /// [xyz] * 3(triangle) * num_faces
  unsigned int *faces;               /// triangle x num_faces
  unsigned int *material_ids;        /// index x num_faces
} Mesh;

void calcNormal(double3 &N, double3 v0, double3 v1, double3 v2) {
  double3 v10 = v1 - v0;
  double3 v20 = v2 - v0;

  N = vcross(v20, v10);
  N.normalize();
}


bool LoadObj(Mesh &mesh, std::vector<tinyobj::material_t> &materials,
             const char *filename, double scale, const char *mtl_path) {
  std::vector<tinyobj::shape_t> shapes;

  std::string err = tinyobj::LoadObj(shapes, materials, filename, mtl_path);

  if (!err.empty()) {
    std::cerr << err << std::endl;
    return false;
  }

  std::cout << "[LoadOBJ] # of shapes in .obj : " << shapes.size() << std::endl;
  std::cout << "[LoadOBJ] # of materials in .obj : " << materials.size()
            << std::endl;

  size_t num_vertices = 0;
  size_t num_faces = 0;
  for (size_t i = 0; i < shapes.size(); i++) {
    printf("  shape[%ld].name = %s\n", i, shapes[i].name.c_str());
    printf("  shape[%ld].indices: %ld\n", i, shapes[i].mesh.indices.size());
    assert((shapes[i].mesh.indices.size() % 3) == 0);
    printf("  shape[%ld].vertices: %ld\n", i, shapes[i].mesh.positions.size());
    assert((shapes[i].mesh.positions.size() % 3) == 0);
    printf("  shape[%ld].normals: %ld\n", i, shapes[i].mesh.normals.size());
    assert((shapes[i].mesh.normals.size() % 3) == 0);

    num_vertices += shapes[i].mesh.positions.size() / 3;
    num_faces += shapes[i].mesh.indices.size() / 3;
  }
  std::cout << "[LoadOBJ] # of faces: " << num_faces << std::endl;
  std::cout << "[LoadOBJ] # of vertices: " << num_vertices << std::endl;

  // @todo { material and texture. }

  // Shape -> Mesh
  mesh.num_faces = num_faces;
  mesh.num_vertices = num_vertices;
  mesh.vertices = new double[num_vertices * 3];
  mesh.faces = new unsigned int[num_faces * 3];
  mesh.material_ids = new unsigned int[num_faces];
  memset(mesh.material_ids, 0, sizeof(int) * num_faces);
  mesh.facevarying_normals = new double[num_faces * 3 * 3];
  mesh.facevarying_uvs = new double[num_faces * 3 * 2];
  memset(mesh.facevarying_uvs, 0, sizeof(double) * 2 * 3 * num_faces);

  // @todo {}
  mesh.facevarying_tangents = NULL;
  mesh.facevarying_binormals = NULL;

  size_t vertexIdxOffset = 0;
  size_t faceIdxOffset = 0;
  for (size_t i = 0; i < shapes.size(); i++) {
    for (size_t f = 0; f < shapes[i].mesh.indices.size() / 3; f++) {
      mesh.faces[3 * (faceIdxOffset + f) + 0] =
          shapes[i].mesh.indices[3 * f + 0];
      mesh.faces[3 * (faceIdxOffset + f) + 1] =
          shapes[i].mesh.indices[3 * f + 1];
      mesh.faces[3 * (faceIdxOffset + f) + 2] =
          shapes[i].mesh.indices[3 * f + 2];

      mesh.faces[3 * (faceIdxOffset + f) + 0] += vertexIdxOffset;
      mesh.faces[3 * (faceIdxOffset + f) + 1] += vertexIdxOffset;
      mesh.faces[3 * (faceIdxOffset + f) + 2] += vertexIdxOffset;

      mesh.material_ids[faceIdxOffset + f] = shapes[i].mesh.material_ids[f];
    }

    for (size_t v = 0; v < shapes[i].mesh.positions.size() / 3; v++) {
      mesh.vertices[3 * (vertexIdxOffset + v) + 0] =
          scale * shapes[i].mesh.positions[3 * v + 0];
      mesh.vertices[3 * (vertexIdxOffset + v) + 1] =
          scale * shapes[i].mesh.positions[3 * v + 1];
      mesh.vertices[3 * (vertexIdxOffset + v) + 2] =
          scale * shapes[i].mesh.positions[3 * v + 2];
    }

    if (shapes[i].mesh.normals.size() > 0) {
      for (size_t f = 0; f < shapes[i].mesh.indices.size() / 3; f++) {
        int f0, f1, f2;

        f0 = shapes[i].mesh.indices[3 * f + 0];
        f1 = shapes[i].mesh.indices[3 * f + 1];
        f2 = shapes[i].mesh.indices[3 * f + 2];

        double3 n0, n1, n2;

        n0[0] = shapes[i].mesh.normals[3 * f0 + 0];
        n0[1] = shapes[i].mesh.normals[3 * f0 + 1];
        n0[2] = shapes[i].mesh.normals[3 * f0 + 2];

        n1[0] = shapes[i].mesh.normals[3 * f1 + 0];
        n1[1] = shapes[i].mesh.normals[3 * f1 + 1];
        n1[2] = shapes[i].mesh.normals[3 * f1 + 2];

        n2[0] = shapes[i].mesh.normals[3 * f2 + 0];
        n2[1] = shapes[i].mesh.normals[3 * f2 + 1];
        n2[2] = shapes[i].mesh.normals[3 * f2 + 2];

        mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 0) + 0] = n0[0];
        mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 0) + 1] = n0[1];
        mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 0) + 2] = n0[2];

        mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 1) + 0] = n1[0];
        mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 1) + 1] = n1[1];
        mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 1) + 2] = n1[2];

        mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 2) + 0] = n2[0];
        mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 2) + 1] = n2[1];
        mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 2) + 2] = n2[2];
      }
    } else {
      // calc geometric normal
      for (size_t f = 0; f < shapes[i].mesh.indices.size() / 3; f++) {
        int f0, f1, f2;

        f0 = shapes[i].mesh.indices[3 * f + 0];
        f1 = shapes[i].mesh.indices[3 * f + 1];
        f2 = shapes[i].mesh.indices[3 * f + 2];

        double3 v0, v1, v2;

        v0[0] = shapes[i].mesh.positions[3 * f0 + 0];
        v0[1] = shapes[i].mesh.positions[3 * f0 + 1];
        v0[2] = shapes[i].mesh.positions[3 * f0 + 2];

        v1[0] = shapes[i].mesh.positions[3 * f1 + 0];
        v1[1] = shapes[i].mesh.positions[3 * f1 + 1];
        v1[2] = shapes[i].mesh.positions[3 * f1 + 2];

        v2[0] = shapes[i].mesh.positions[3 * f2 + 0];
        v2[1] = shapes[i].mesh.positions[3 * f2 + 1];
        v2[2] = shapes[i].mesh.positions[3 * f2 + 2];

        double3 N;
        calcNormal(N, v0, v1, v2);

        mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 0) + 0] = N[0];
        mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 0) + 1] = N[1];
        mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 0) + 2] = N[2];

        mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 1) + 0] = N[0];
        mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 1) + 1] = N[1];
        mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 1) + 2] = N[2];

        mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 2) + 0] = N[0];
        mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 2) + 1] = N[1];
        mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 2) + 2] = N[2];
      }
    }

    if (shapes[i].mesh.texcoords.size() > 0) {
      for (size_t f = 0; f < shapes[i].mesh.indices.size() / 3; f++) {
        int f0, f1, f2;

        f0 = shapes[i].mesh.indices[3 * f + 0];
        f1 = shapes[i].mesh.indices[3 * f + 1];
        f2 = shapes[i].mesh.indices[3 * f + 2];

        double3 n0, n1, n2;

        n0[0] = shapes[i].mesh.texcoords[2 * f0 + 0];
        n0[1] = shapes[i].mesh.texcoords[2 * f0 + 1];

        n1[0] = shapes[i].mesh.texcoords[2 * f1 + 0];
        n1[1] = shapes[i].mesh.texcoords[2 * f1 + 1];

        n2[0] = shapes[i].mesh.texcoords[2 * f2 + 0];
        n2[1] = shapes[i].mesh.texcoords[2 * f2 + 1];

        mesh.facevarying_uvs[2 * (3 * (faceIdxOffset + f) + 0) + 0] = n0[0];
        mesh.facevarying_uvs[2 * (3 * (faceIdxOffset + f) + 0) + 1] = n0[1];

        mesh.facevarying_uvs[2 * (3 * (faceIdxOffset + f) + 1) + 0] = n1[0];
        mesh.facevarying_uvs[2 * (3 * (faceIdxOffset + f) + 1) + 1] = n1[1];

        mesh.facevarying_uvs[2 * (3 * (faceIdxOffset + f) + 2) + 0] = n2[0];
        mesh.facevarying_uvs[2 * (3 * (faceIdxOffset + f) + 2) + 1] = n2[1];
      }
    }

    vertexIdxOffset += shapes[i].mesh.positions.size() / 3;
    faceIdxOffset += shapes[i].mesh.indices.size() / 3;
  }

  return true;
}
}  // namespace

int main(int argc, char **argv) {
  int width = 8192;
  int height = 8192;

  double scale = 1.0;

  std::string objFilename = "../common/cornellbox_suzanne_lucy.obj";
  std::string mtlPath = "../common/";

  if (argc > 1) {
    objFilename = std::string(argv[1]);
  }

  if (argc > 2) {
    scale = atof(argv[2]);
  }

  if (argc > 3) {
    mtlPath = std::string(argv[3]);
  }

  bool ret = false;

  Mesh mesh;
  std::vector<tinyobj::material_t> materials;
  ret = LoadObj(mesh, materials, objFilename.c_str(), scale, mtlPath.c_str());
  if (!ret) {
    fprintf(stderr, "Failed to load [ %s ]\n", objFilename.c_str());
    return -1;
  }

  nanort::BVHBuildOptions<double> build_options;  // Use default option
  build_options.cache_bbox = false;

  printf("  BVH build option:\n");
  printf("    # of leaf primitives: %d\n", build_options.min_leaf_primitives);
  printf("    SAH binsize         : %d\n", build_options.bin_size);

  nanort::TriangleMesh<double> triangle_mesh(mesh.vertices, mesh.faces, sizeof(double) * 3);
  nanort::TriangleSAHPred<double> triangle_pred(mesh.vertices, mesh.faces, sizeof(double) * 3);

  printf("num_triangles = %lu\n", mesh.num_faces);
  printf("faces = %p\n", mesh.faces);

  nanort::BVHAccel<double> accel;
  ret = accel.Build(mesh.num_faces, triangle_mesh, triangle_pred, build_options);
  assert(ret);

  nanort::BVHBuildStatistics stats = accel.GetStatistics();

  printf("  BVH statistics:\n");
  printf("    # of leaf   nodes: %d\n", stats.num_leaf_nodes);
  printf("    # of branch nodes: %d\n", stats.num_branch_nodes);
  printf("  Max tree depth     : %d\n", stats.max_tree_depth);
  double bmin[3], bmax[3];
  accel.BoundingBox(bmin, bmax);
  printf("  Bmin               : %f, %f, %f\n", bmin[0], bmin[1], bmin[2]);
  printf("  Bmax               : %f, %f, %f\n", bmax[0], bmax[1], bmax[2]);


#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {

        double3 rayDir = double3((x / (double)width) - 0.5,
                               (y / (double)height) - 0.5, -1.0);
        rayDir.normalize();

        double3 rayOrg = double3(0.0, 5.0, 20.0);


          nanort::Ray<double> ray;
          double kFar = 1.0e+30;
          ray.min_t = 0.001;
          ray.max_t = kFar;

          ray.dir[0] = rayDir[0];
          ray.dir[1] = rayDir[1];
          ray.dir[2] = rayDir[2];
          ray.org[0] = rayOrg[0];
          ray.org[1] = rayOrg[1];
          ray.org[2] = rayOrg[2];

          nanort::TriangleIntersector<double, nanort::TriangleIntersection<double>> triangle_intersector(
              mesh.vertices, mesh.faces, sizeof(double) * 3);
          nanort::TriangleIntersection<double> isect;
          bool hit = accel.Traverse(ray, triangle_intersector, &isect);
    }
  }
  return 0;
}
