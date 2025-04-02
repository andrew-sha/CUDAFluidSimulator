#include "platformgl.h"
#include <cuda_gl_interop.h>

GLuint vbo;  // OpenGL Vertex Buffer Object (will store particle information)
cudaGraphicsResource *cudaVBO; // CUDA resource

// OpenGL rendering function
void display() {
    glClear(GL_COLOR_BUFFER_BIT);

    // Map VBO to CUDA
    float2 *d_pos;
    size_t size;
    cudaGraphicsMapResources(1, &cudaVBO, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&d_pos, &size, cudaVBO);

    // Launch CUDA kernel

    cudaDeviceSynchronize();

    // Unmap VBO
    cudaGraphicsUnmapResources(1, &cudaVBO, 0);

    // Render using OpenGL
    glEnableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(2, GL_FLOAT, 0, 0);
    glDrawArrays(GL_POINTS, 0, numParticles);
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();
    glutPostRedisplay();
}

// Setup OpenGL and CUDA buffer
void initGLCUDA() {
    // Create OpenGL vertex buffer
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, numParticles * sizeof(float2), NULL, GL_DYNAMIC_DRAW);

    // Register VBO with CUDA
    cudaGraphicsGLRegisterBuffer(&cudaVBO, vbo, cudaGraphicsMapFlagsWriteDiscard);
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(800, 600);
    glutCreateWindow("CUDA SPH Simulation");

    initGLCUDA();

    glutDisplayFunc(display);
    glutMainLoop();

    cudaGraphicsUnregisterResource(cudaVBO);
    glDeleteBuffers(1, &vbo);
    return 0;
}
