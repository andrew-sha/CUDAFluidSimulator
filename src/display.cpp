#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>

#include "platformgl.h"
#include "simulator.h"

float boxVertices[8][3] = {{0.0f, 0.0f, 0.0f},    {10.0f, 0.0f, 0.0f},
                           {10.0f, 10.0f, 0.0f},  {0.0f, 10.0f, 0.0f},
                           {0.0f, 0.0f, 10.0f},   {10.0f, 0.0f, 10.0f},
                           {10.0f, 10.0f, 10.0f}, {0.0f, 10.0f, 10.0f}};

int boxEdges[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6},
                       {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};

Simulator *simulator = NULL;
bool mouseClicked = false;
int2 clickCoords;

void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        printf("Left click at (%d, %d)\n", x, y);

        if ((x < BOX_MIN_X) || (x >= BOX_MAX_X) || (y < BOX_MIN_Y) ||
            (y >= BOX_MAX_Y)) {
            return; // Out of bounds click
        }

        mouseClicked = true;
        clickCoords = make_int2(x, y);
    }
}

// OpenGL rendering function
void display() {
    simulator->simulate();
    const float3 *positions = simulator->getPosition();

    auto renderStart = std::chrono::steady_clock::now();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    // Draw the box edges
    glColor3f(1.0f, 1.0f, 1.0f);
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

    if (mouseClicked) {
        mouseClicked = false;
    }
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
    glTranslatef(-5.0f, -5.0f, -15.0f); // Move the camera back along Z axis
    glMatrixMode(GL_MODELVIEW);

    glutDisplayFunc(display);
    glutMouseFunc(mouse);

    glutMainLoop();
}
