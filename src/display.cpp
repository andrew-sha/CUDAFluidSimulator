#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <cuda_runtime.h>


#include "platformgl.h"
#include "simulator.h"

float boxVertices[8][3] = {{0.0f, 0.0f, 0.0f},    {10.0f, 0.0f, 0.0f},
                           {10.0f, 10.0f, 0.0f},  {0.0f, 10.0f, 0.0f},
                           {0.0f, 0.0f, 10.0f},   {10.0f, 0.0f, 10.0f},
                           {10.0f, 10.0f, 10.0f}, {0.0f, 10.0f, 10.0f}};

int boxEdges[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6},
                       {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};

Simulator *simulator = NULL;

// OpenGL rendering function
void handleDisplay() {
    // Might want to wrap this in timing code for perf measurement
    // In general, we need to plan out how we will measure performance
    // for now using the setup to test rendering different frame each iteration
    simulator->simulate();
    const float3 *positions = simulator->getPosition();


    auto renderStart = std::chrono::steady_clock::now();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear the screen
    glLoadIdentity();                                   // Reset transformations

    glTranslatef(-5.f, -5.f, -15.0f); // Move the camera back along Z axis

    // Draw the box edges
    glColor3f(1.0f, 1.0f, 1.0f); // White
    glBegin(GL_LINES);
    for (int i = 0; i < 12; i++) {
        glVertex3fv(boxVertices[boxEdges[i][0]]);
        glVertex3fv(boxVertices[boxEdges[i][1]]);
    }
    glEnd();

    // Render each particle as a point
    glColor3f(0.0f, 0.0f, 1.0f);
    glBegin(GL_POINTS);
    for (int i = 0; i < simulator->settings->numParticles; i++) {
        glVertex3f(positions[i].x, positions[i].y, positions[i].z);
    }
    glEnd();
    
    const double renderTime = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - renderStart).count();
    std::cout << "Render time (sec): " << std::fixed << std::setprecision(10) << renderTime << std::endl;

    glutSwapBuffers();
    glutPostRedisplay();
}

void startVisualization(Simulator *sim) {
    // Declare simulator instance
    simulator = sim;

    // Initialize glut
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(800, 600);
    glutCreateWindow("SPH Simulation");

    // Initialize gl
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Set background color to black
    glEnable(GL_POINT_SMOOTH);            // Enable point smoothing (optional)
    glPointSize(3.0f);                    // Set point size for particles
    glEnable(GL_DEPTH_TEST);              // Enable depth testing for 3D

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(-2.0, 2.0, -2.0, 2.0, 1.0, 100.0);
    glMatrixMode(GL_MODELVIEW);

    glutDisplayFunc(handleDisplay);
    glutMainLoop();
}
