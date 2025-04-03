#include <algorithm>

#include "platformgl.h"
#include "simulator.h"

Simulator* simulator = NULL;

// OpenGL rendering function
void
handleDisplay() {
    // Might want to wrap this in timing code for perf measurement
    // In general, we need to plan out how we will measure performance
    simulator->simulate();

    const float* positions = simulator->getPosition();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  // Clear the screen
    glLoadIdentity();                                  // Reset transformations

    glTranslatef(0.0f, 0.0f, -5.0f);  // Move the camera back along Z axis

    // Render each particle as a point
    glBegin(GL_POINTS);
    for (size_t i = 0; i < simulator->settings->numParticles; i++) {
        size_t particleIdx = 3 * i;
        glVertex3f(positions[particleIdx], positions[particleIdx+1], positions[particleIdx+2]);
    }
    glEnd();

    glutSwapBuffers();
    glutPostRedisplay();
}

void startVisualization(Simulator* sim) {
    // Declare simulator instance
    simulator = sim;

    // Initialize glut
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(800, 600);
    glutCreateWindow("CUDA SPH Simulation");

    // Initialize gl
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);  // Set background color to black
    glEnable(GL_POINT_SMOOTH);               // Enable point smoothing (optional)
    glPointSize(5.0f);                       // Set point size for particles
    glEnable(GL_DEPTH_TEST);                 // Enable depth testing for 3D

    glutDisplayFunc(handleDisplay);
    glutMainLoop();
}
