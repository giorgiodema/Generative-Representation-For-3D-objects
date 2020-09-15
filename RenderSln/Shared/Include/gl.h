#ifndef GL_H_INCLUDED
#define GL_H_INCLUDED


#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <vector>
#include <iostream>
#include "shader.h"
#include "model.h"

using namespace glm;
using namespace std;

struct Constants {
	// Screen
	int SCREEN_WIDTH;
	int SCREEN_HEIGHT;

	// Camera
	vec3 CAMERA_POS;
	vec3 CAMERA_AT;
	vec3 CAMERA_UP;

	// Light
	vec3 LIGHT_POS;
	vec3 LIGHT_COLOR;
	float AMBIENT_STRENGHT;

	// Vexel
	vec3 VOXEL_COLOR;
};

extern Constants consts;

int InitializeGLContext(GLFWwindow** window);

// Light Transformations
void RotateLight(Shader* s, vec3 axis, float angle);

class GraphicObject {
	vector<GLfloat>* vertices;
	vector<GLfloat>* normals;
	vector<GLfloat>* colors;
	vector<GLint>* indices;
	Model* objModel;
	GLuint vertexbuffer;
	GLuint normalsbuffer;
	GLuint colorsbuffer;
	GLuint elementbuffer;
	mat4 model;
	mat4 view;
	mat4 projection;
	const static int VERTEX_LAYOUT = 0;
	const static int NORMAL_LAYOUT = 1;
	const static int COLOR_LAYOUT = 2;
	const static int ANGLE = 1;

	// functions to update gl context
	void bufferVertices();
	void bindVertices();
	void bufferNormals();
	void bindNormals();
	void bufferColors();
	void bindColors();
	void bufferIndices();
	void bindIndices();
public:
	// constructors
	GraphicObject(vector<GLfloat>* vertices);
	GraphicObject(vector<GLfloat>* vertices, vector<GLfloat>* normals);
	GraphicObject(vector<GLfloat>* vertices, vector<GLfloat>* normals, vector<GLfloat>* colors);
	GraphicObject(Model* objModel);

	// geometric transformations
	void rotate(vec3 axis, float degree);
	void scale(float factor);
	void translate(vec3 v);

	// render
	void render(Shader* s);

	// set the camera
	void setCamera();
};

std::vector<float>* cubeVertices();
std::vector<float>* cubeNormals();

#endif
