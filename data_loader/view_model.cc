// Copyright (C) 2014 The Regents of the University of California (Regents).
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of The Regents or University of California nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Chris Sweeney (cmsweeney@cs.ucsb.edu)

#include <Eigen/Core>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <theia/theia.h>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <OpenGL/OpenGL.h>
#ifdef FREEGLUT
#include <GL/freeglut.h>
#else
#include <GLUT/glut.h>
#endif
#else
#ifdef _WIN32
#include <windows.h>
#endif
#define GL_GLEXT_PROTOTYPES 1
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#endif

DEFINE_string(model, "", "Model file to be viewe.");

// Containers for the data.
std::vector<theia::Camera> cameras;
std::vector<Eigen::Vector3d> world_points;

// Parameters for OpenGL.
int mouse_down_x[3], mouse_down_y[3];
float rot_x = 0.0f, rot_y = 0.0f;
float distance = 100.0;
Eigen::Vector3d origin = Eigen::Vector3d::Zero();
bool mouse_rotates = false;
bool draw_cameras = true;

float point_size = 1.0;
float camera_scale = 1.0;

void GetPerspectiveParams(double* aspect_ratio, double* fovy) {
  int width = 800;
  int height = 600;
  double focal_length = 600.0;
  *aspect_ratio = static_cast<double>(width) / static_cast<double>(height);
  *fovy = 2 * atan(height / (2.0 * focal_length)) * 180.0 / M_PI;
}

void ChangeSize(int w, int h) {
  // Prevent a divide by zero, when window is too short
  // (you cant make a window of zero width).
  if (h == 0) h = 1;

  // Use the Projection Matrix
  glMatrixMode(GL_PROJECTION);

  // Reset Matrix
  glLoadIdentity();

  // Set the viewport to be the entire window
  double aspect_ratio, fovy;
  GetPerspectiveParams(&aspect_ratio, &fovy);
  glViewport(0, 0, w, h);

  // Set the correct perspective.
  gluPerspective(fovy, aspect_ratio, 0.01f, 100000.0f);

  // Get Back to the Modelview
  glMatrixMode(GL_MODELVIEW);
}

void DrawAxes(float length) {
  glPushAttrib(GL_POLYGON_BIT | GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT);

  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  glDisable(GL_LIGHTING);

  glBegin(GL_LINES);
  glColor3f(1.0, 0.0, 0.0);
  glVertex3f(0, 0, 0);
  glVertex3f(length, 0, 0);

  glColor3f(0.0, 1.0, 0.0);
  glVertex3f(0, 0, 0);
  glVertex3f(0, length, 0);

  glColor3f(0.0, 0.0, 1.0);
  glVertex3f(0, 0, 0);
  glVertex3f(0, 0, length);
  glEnd();

  glPopAttrib();
}

void DrawCamera(const theia::Camera& camera) {
  const Eigen::Matrix3d rotation =
      camera.GetOrientationAsRotationMatrix().transpose();
  const Eigen::Vector3d position = camera.GetPosition();

  glPushMatrix();
  // Apply world pose transformation.
  GLdouble glm[16];
  glm[0] = rotation(0, 0);
  glm[1] = rotation(1, 0);
  glm[2] = rotation(2, 0);
  glm[3] = 0.0;
  glm[4] = rotation(0, 1);
  glm[5] = rotation(1, 1);
  glm[6] = rotation(2, 1);
  glm[7] = 0.0;
  glm[8] = rotation(0, 2);
  glm[9] = rotation(1, 2);
  glm[10] = rotation(2, 2);
  glm[11] = 0.0;
  glm[12] = position[0];
  glm[13] = position[1];
  glm[14] = position[2];
  glm[15] = 1.0;
  glMultMatrixd(glm);

  glScalef(camera_scale, camera_scale, camera_scale);

  // Draw Camera
  glColor3f(1.0, 0.0, 0.0);
  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  glBegin(GL_TRIANGLE_FAN);
  glVertex3f(0.0, 0.0, 0.0);
  glVertex3f(1.0, 1.0, 1.0);
  glVertex3f(1.0, -1.0, 1.0);
  glVertex3f(-1.0, -1.0, 1.0);
  glVertex3f(-1.0, 1.0, 1.0);
  glVertex3f(1.0, 1.0, 1.0);
  glEnd();
  glPopMatrix();
}

void RenderScene() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glLoadIdentity();
  glTranslatef(0.0f, 0.0f, -distance);
  glRotatef(180.0f + rot_x, 1.0f, 0.0f, 0.0f);
  glRotatef(-rot_y, 0.0f, 1.0f, 0.0f);
  glTranslatef(-origin[0], -origin[1], -origin[2]);
  glClearColor(1.0f, 1.0f, 1.0f, 0.0f);

  // Plot the point cloud.
  glEnable(GL_MULTISAMPLE);
  glEnable(GL_BLEND);
  glEnable(GL_POINT_SMOOTH);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glPointParameterf(GL_POINT_SIZE_MIN, 0.0);

  glPointSize(point_size);
  glPointParameterf(GL_POINT_SIZE_MIN, 0.1f);
  glPointParameterf(GL_POINT_SIZE_MAX, 8.0f);

  // the coordinates for calcluting point attenuation:
  GLfloat point_size_coords[3];
  point_size_coords[0] = 1.0f;
  point_size_coords[1] = 0.055f;
  point_size_coords[2] = 0.0f;
  glPointParameterfv(GL_POINT_DISTANCE_ATTENUATION, point_size_coords);

  // Draw the points.
  glColor3f(0.0, 0.0, 0.0);
  glBegin(GL_POINTS);
  for (int i = 0; i < world_points.size(); i++) {
    glVertex3d(world_points[i].x(), world_points[i].y(), world_points[i].z());
  }

  // Draw the cameras.
  if (draw_cameras) {
    for (int i = 0; i < cameras.size(); i++) {
      DrawCamera(cameras[i]);
    }
  }
  glFlush();
  glutSwapBuffers();
}

void MouseButton(int button, int state, int x, int y) {
  // button down: save coordinates
  if (state == GLUT_DOWN && button <= 2) {
    mouse_down_x[button] = x;
    mouse_down_y[button] = y;
    if (button == GLUT_LEFT_BUTTON) mouse_rotates = true;
    return;
  }

  if (state == GLUT_UP && button == GLUT_LEFT_BUTTON) {
    mouse_rotates = false;
  }

  // scroll event - wheel reports as button 3 (scroll up) and button 4 (scroll
  // down)
  if ((button == 3) || (button == 4)) {
    // Each wheel event reports like a button click, GLUT_DOWN then GLUT_UP
    if (state == GLUT_UP) return;  // Disregard redundant GLUT_UP events
    if (button == 3)
      distance /= 1.5f;
    else
      distance *= 1.5f;
  }
}

void MouseMove(int x, int y) {
  if (mouse_rotates) {
    rot_x -= (y - mouse_down_y[GLUT_LEFT_BUTTON]) * 0.05f;
    rot_y -= (x - mouse_down_x[GLUT_LEFT_BUTTON]) * 0.05f;
  }
}

void Keyboard(unsigned char key, int x, int y) {
  switch (key) {
    case 'r':  // reset viewpoint
      distance = 100.0f;
      rot_x = 0.0f;
      rot_y = 0.0f;
      point_size = 1.0;

      origin = Eigen::Vector3d::Zero();
      break;
    case 'a':
      distance /= 1.2f;
      break;
    case 'z':
      distance *= 1.2f;
      break;
    case 'p':
      point_size /= 1.2;
      break;
    case 'P':
      point_size *= 1.2;
      break;
    case 'f':
      camera_scale /= 1.2;
      break;
    case 'F':
      camera_scale *= 1.2;
      break;
    case 'u':
      origin[0] += 10.0;
      break;
    case 'U':
      origin[0] -= 10.0;
      break;
    case 'i':
      origin[2] += 10.0;
      break;
    case 'I':
      origin[2] -= 10.0;
      break;
    case 'c':
      draw_cameras = !draw_cameras;
      break;
  }
}

// initialize viewport etc.
void Init() {
  glMatrixMode(GL_PROJECTION);

  glLoadIdentity();

  double aspect_ratio, fovy;  // set the correct perspective.
  GetPerspectiveParams(&aspect_ratio, &fovy);
  gluPerspective(fovy, 1.0, .01, 100000.0);

  glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  // Output as a binary file.
  std::unique_ptr<theia::Model> model(new theia::Model());
  CHECK(ReadModel(FLAGS_model, model.get()))
      << "Could not read model file.";

  // Set up world points.
  world_points.reserve(model->NumTracks());
  for (const theia::TrackId track_id : model->TrackIds()) {
    const auto* track = model->Track(track_id);
    if (track == nullptr || !track->IsEstimated()) {
      continue;
    }
    world_points.emplace_back(track->Point().hnormalized());
  }

  // Set up camera drawing.
  cameras.reserve(model->NumViews());
  for (const theia::ViewId view_id : model->ViewIds()) {
    const auto* view = model->View(view_id);
    if (view == nullptr || !view->IsEstimated()) {
      continue;
    }
    cameras.emplace_back(view->Camera());
  }

  model.release();

  // Set up opengl and glut.
  glutInit(&argc, argv);
  glutInitWindowPosition(600, 600);
  glutInitWindowSize(1200, 800);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  glutCreateWindow("Bundler Reconstruction Viewer");

  // Set the camera
  gluLookAt(0.0f, 0.0f, -6.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);

  // register callbacks
  glutDisplayFunc(RenderScene);
  glutReshapeFunc(ChangeSize);
  glutMouseFunc(MouseButton);
  glutMotionFunc(MouseMove);
  glutKeyboardFunc(Keyboard);
  glutIdleFunc(RenderScene);

  // enter GLUT event processing loop
  glutMainLoop();

  return 0;
}
