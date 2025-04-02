#include "platformgl.h"
#include "simulator.h"

Simulator* simulator = NULL;

// OpenGL rendering function
void
handleDisplay(Simulator* simulator) {

    // simulation and rendering work is done in the renderPicture
    // function below

    // probably wrap this in timing code for perf measurement
    simulator->simulate();

    float* positions = simulator->getPosition();

    // the subsequent code will OpenGL to present the new position of the particles on screen
    // RIGHT NOW, THE CODE IS WRONG, ITS JUST A PLACEHOLDER TO SHOW THE SHAPE OF THE LOGIC
    int width = std::min(img->width, gDisplay.width);
    int height = std::min(img->height, gDisplay.height);

    glDisable(GL_DEPTH_TEST);
    glClearColor(0.f, 0.f, 0.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.f, gDisplay.width, 0.f, gDisplay.height, -1.f, 1.f);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // copy image data from the renderer to the OpenGL
    // frame-buffer.  This is inefficient solution is the processing
    // to generate the image is done in CUDA.  An improved solution
    // would render to a CUDA surface object (stored in GPU memory),
    // and then bind this surface as a texture enabling it's use in
    // normal openGL rendering
    glRasterPos2i(0, 0);
    glDrawPixels(width, height, GL_RGBA, GL_FLOAT, img->data);

    double currentTime = CycleTimer::currentSeconds();

    if (gDisplay.printStats)
        printf("%.2f ms\n", 1000.f * (currentTime - gDisplay.lastFrameTime));

    gDisplay.lastFrameTime = currentTime;

    glutSwapBuffers();
    glutPostRedisplay();
}

void startVisualization(Simulator* sim) {
    // Declare simulator instance
    simulator = sim;

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(800, 600);
    glutCreateWindow("CUDA SPH Simulation");


    glutDisplayFunc(handleDisplay);
    glutMainLoop();
}
