#include <string>
#include <iostream>
#include <sstream> //std::stringstream
#include <vector>
#include <windows.h>
#include "gl.h"
#include "loaders.h"

using namespace std;
using namespace glm;

int STEPS = 720;
int LIGHT_STEPS = 8;
vec3 AXIS = vec3(0, 0, 0);
bool ROTATE = false;

void beforeRender(GraphicObject* o,vec3 axis,float angle) {
	
	if(ROTATE)
		o->rotate(axis, angle);
	o->translate(vec3(-0.5, -0.5, -0.5));
	o->scale(3.0 / 2.0);

}

void afterRender(GraphicObject* o) {
	o->scale(2.0 / 3.0);
	o->translate(vec3(0.5, 0.5, 0.5));
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_X || key == GLFW_KEY_Y || key == GLFW_KEY_Z) {
		if (action == GLFW_PRESS) {
			ROTATE = true;
			switch (key)
			{
			case GLFW_KEY_X:
				AXIS = vec3(1, 0, 0);
				break;
			case GLFW_KEY_Y:
				AXIS = vec3(0, 1, 0);
				break;
			case GLFW_KEY_Z:
				AXIS = vec3(0, 0, 1);
				break;
			default:
				break;
			};
		}
		if (action == GLFW_RELEASE) {
			ROTATE = false;
		}
	}
}

int main(int argc,char**argv)
{

	if (argc != 2 && argc != 3) {
		cout << "usage: ./" << argv[0] << " path1.binvox [path2.binvox]";
		return -1;
	}

	char* path1 = argv[1];
	char* path2 = argc == 3 ? argv[2] : NULL;

	/* Initialize Context */
	GLFWwindow* window;
	if (InitializeGLContext(&window) < 0) {
		std::cout << "Falied to initialize GL context";
		return -1;
	}

	/* Create the shaders */
	Shader* voxel_shader = new Shader("vertexShader.vert", "fragmentShader.frag");
	Shader* light_shader = new Shader("lightvs.vert", "lightfs.frag");

	/* Set the parameters of each program */
	light_shader->use();
	light_shader->setVec3("v_light_color", consts.LIGHT_COLOR);

	voxel_shader->use();
	voxel_shader->setVec3("v_light_color", consts.LIGHT_COLOR);
	voxel_shader->setVec3("v_voxel_color", consts.VOXEL_COLOR);
	voxel_shader->setFloat("v_ambient_strenght", consts.AMBIENT_STRENGHT);
	voxel_shader->setVec3("v_light_pos", consts.LIGHT_POS);

	/* Set background color */
	glClearColor(0.8f, 0.8f, 0.8f, 1.0f);
	// Enable depth test
	glEnable(GL_DEPTH_TEST);
	// Accept fragment if it closer to the camera than the former one
	//glDepthFunc(GL_LESS);

	/* Initialize VAO */
	GLuint VertexArrayID;
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);

	/* create voxelloader */
	BVoxLoader bl1 = BVoxLoader{ path1 };
	GraphicObject bo1 = { bl1.getVertices(),bl1.getNormals()};

	BVoxLoader* bl2 = path2 ? new BVoxLoader(path2) : NULL;
	GraphicObject* bo2 = bl2 ? new GraphicObject(bl2->getVertices(), bl2->getNormals()) : NULL;
	delete bl2;

	/* create light object*/
	vector<float>* cubev = cubeVertices();
	GraphicObject light_obj = { cubev };
	light_obj.scale(0.05);
	light_obj.translate(consts.LIGHT_POS);

	int step = 1;
	int n_axis = 0;
	float angle = 6.28318530718 / STEPS;


	/* Loop until the user closes the window */
	while (!glfwWindowShouldClose(window))
	{

		// Clear the screen
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		if(argc==3)
			glViewport(0, consts.SCREEN_HEIGHT / 4, consts.SCREEN_WIDTH / 2, consts.SCREEN_HEIGHT / 2);

		vec3 axis = vec3(n_axis == 0 ? 1 : 0, n_axis == 1 ? 1 : 0, n_axis == 2 ? 1 : 0);
		beforeRender(&bo1,AXIS,angle);
		bo1.render(voxel_shader);
		afterRender(&bo1);

		if (argc == 3) {
			glViewport(consts.SCREEN_WIDTH/2, consts.SCREEN_HEIGHT / 4, consts.SCREEN_WIDTH / 2, consts.SCREEN_HEIGHT / 2);
			beforeRender(bo2, AXIS, angle);
			bo2->render(voxel_shader);
			afterRender(bo2);
		}

		//light_obj.render(light_shader);

		//
		// set kye events
		//
		glfwSetKeyCallback(window, key_callback);

		/* Swap front and back buffers */
		glfwSwapBuffers(window);

		/* Poll for and process events */
		glfwPollEvents();
		//Sleep(1000);
	}

	glfwTerminate();
	return 0;
}