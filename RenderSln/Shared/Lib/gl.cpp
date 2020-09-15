#include "gl.h"

Constants consts = {
	1440,				// Screen Width
	1080,				// Screen Height
	vec3(0, 1, 3),		// Camera Pos
	vec3(0, 0, 0),		// Camera at
	vec3(0, 1, 0),		// Camera Up
	vec3(0, 10, 10),		// Light Pos
	vec3(1.0, 1.0, 1.0),// Light Color
	0.3,				// Ambient Strenght
	vec3(1.0, 0.0, 0.0)	// Voxel Color
};

int InitializeGLContext(GLFWwindow** window) {
	/* Initialize GLEW */
	if (!glfwInit())
		return -1;


	/* Create a windowed mode window and its OpenGL context */
	GLFWwindow* w;
	w = glfwCreateWindow(consts.SCREEN_WIDTH, consts.SCREEN_HEIGHT, "Hello World", NULL, NULL);
	if (!w)
	{
		glfwTerminate();
		return -1;
	}

	/* Make the window's context current */
	glfwMakeContextCurrent(w);

	/* Initialize glad */
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize OpenGL context" << std::endl;
		return -1;
	}

	printf("OpenGL loaded\n");

	printf("Vendor:          %s\n", glGetString(GL_VENDOR));
	printf("Renderer:        %s\n", glGetString(GL_RENDERER));
	printf("Version OpenGL:  %s\n", glGetString(GL_VERSION));
	printf("Version GLSL:    %s\n\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	*window = w;
	return 0;
};
std::vector<float>* cubeVertices() {
	return new std::vector<float>{
	-0.5f, -0.5f, -0.5f,
	 0.5f, -0.5f, -0.5f,
	 0.5f,  0.5f, -0.5f,
	 0.5f,  0.5f, -0.5f,
	-0.5f,  0.5f, -0.5f,
	-0.5f, -0.5f, -0.5f,

	-0.5f, -0.5f,  0.5f,
	 0.5f, -0.5f,  0.5f,
	 0.5f,  0.5f,  0.5f,
	 0.5f,  0.5f,  0.5f,
	-0.5f,  0.5f,  0.5f,
	-0.5f, -0.5f,  0.5f,

	-0.5f,  0.5f,  0.5f,
	-0.5f,  0.5f, -0.5f,
	-0.5f, -0.5f, -0.5f,
	-0.5f, -0.5f, -0.5f,
	-0.5f, -0.5f,  0.5f,
	-0.5f,  0.5f,  0.5f,

	 0.5f,  0.5f,  0.5f,
	 0.5f,  0.5f, -0.5f,
	 0.5f, -0.5f, -0.5f,
	 0.5f, -0.5f, -0.5f,
	 0.5f, -0.5f,  0.5f,
	 0.5f,  0.5f,  0.5f,

	-0.5f, -0.5f, -0.5f,
	 0.5f, -0.5f, -0.5f,
	 0.5f, -0.5f,  0.5f,
	 0.5f, -0.5f,  0.5f,
	-0.5f, -0.5f,  0.5f,
	-0.5f, -0.5f, -0.5f,

	-0.5f,  0.5f, -0.5f,
	 0.5f,  0.5f, -0.5f,
	 0.5f,  0.5f,  0.5f,
	 0.5f,  0.5f,  0.5f,
	-0.5f,  0.5f,  0.5f,
	-0.5f,  0.5f, -0.5f
	};
};

std::vector<float>* cubeNormals() {
	return new std::vector<float>{
		0.0f,0.0f,1.0f,
		0.0f,0.0f,1.0f,
		0.0f,0.0f,1.0f,
		0.0f,0.0f,1.0f,
		0.0f,0.0f,1.0f,
		0.0f,0.0f,1.0f,

		0.0f,0.0f,-1.0f,
		0.0f,0.0f,-1.0f,
		0.0f,0.0f,-1.0f,
		0.0f,0.0f,-1.0f,
		0.0f,0.0f,-1.0f,
		0.0f,0.0f,-1.0f,

		1.0f,0.0f,0.0f,
		1.0f,0.0f,0.0f,
		1.0f,0.0f,0.0f,
		1.0f,0.0f,0.0f,
		1.0f,0.0f,0.0f,
		1.0f,0.0f,0.0f,

		-1.0f,0.0f,0.0f,
		-1.0f,0.0f,0.0f,
		-1.0f,0.0f,0.0f,
		-1.0f,0.0f,0.0f,
		-1.0f,0.0f,0.0f,
		-1.0f,0.0f,0.0f,

		0.0f,1.0f,0.0f,
		0.0f,1.0f,0.0f,
		0.0f,1.0f,0.0f,
		0.0f,1.0f,0.0f,
		0.0f,1.0f,0.0f,
		0.0f,1.0f,0.0f,

		0.0f,-1.0f,0.0f,
		0.0f,-1.0f,0.0f,
		0.0f,-1.0f,0.0f,
		0.0f,-1.0f,0.0f,
		0.0f,-1.0f,0.0f,
		0.0f,-1.0f,0.0f
	};
};

void RotateLight(Shader* s, vec3 axis, float angle) {
	vec3 new_pos = rotate(consts.LIGHT_POS, angle, axis);
	consts.LIGHT_POS = new_pos;
	s->setVec3("v_light_pos", new_pos);
}

// functions to update gl context
void GraphicObject::bufferVertices() {
	/*	create vertexbuffer on the gpu and send
		data to GPU
	*/
	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, vertices->size() * sizeof(GLfloat), vertices->data(), GL_STATIC_DRAW);

}
void GraphicObject::bindVertices() {
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	// set attributes
	glEnableVertexAttribArray(VERTEX_LAYOUT);
	glVertexAttribPointer(
		VERTEX_LAYOUT,      // attribute 0. No particular reason for 0, but must match the layout in the shader.
		3,                  // size
		GL_FLOAT,           // type
		GL_FALSE,           // normalized?
		0,                  // stride
		(void*)0            // array buffer offset
	);
	// Bind buffer and set attributes
}
void GraphicObject::bufferNormals() {
	glGenBuffers(1, &normalsbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, normalsbuffer);
	glBufferData(GL_ARRAY_BUFFER, normals->size() * sizeof(GLfloat), normals->data(), GL_STATIC_DRAW);
};

void GraphicObject::bindNormals() {
	glBindBuffer(GL_ARRAY_BUFFER, normalsbuffer);
	glEnableVertexAttribArray(NORMAL_LAYOUT);
	glVertexAttribPointer(
		NORMAL_LAYOUT,      // attribute 0. No particular reason for 0, but must match the layout in the shader.
		3,                  // size
		GL_FLOAT,           // type
		GL_FALSE,           // normalized?
		0,                  // stride
		(void*)0            // array buffer offset
	);
};

void GraphicObject::bufferColors() {
	glGenBuffers(2, &colorsbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, colorsbuffer);
	glBufferData(GL_ARRAY_BUFFER, colors->size() * sizeof(GLfloat), colors->data(), GL_STATIC_DRAW);
};

void GraphicObject::bindColors() {
	glBindBuffer(GL_ARRAY_BUFFER, colorsbuffer);
	glEnableVertexAttribArray(COLOR_LAYOUT);
	glVertexAttribPointer(
		COLOR_LAYOUT,      // attribute 2. No particular reason for 2, but must match the layout in the shader.
		3,                  // size
		GL_FLOAT,           // type
		GL_FALSE,           // normalized?
		0,                  // stride
		(void*)0            // array buffer offset
	);
};
void GraphicObject::bufferIndices() {
	/*	create elementbuffer on the gpu and send
		data to gpu
	*/
	glGenBuffers(1, &elementbuffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices->size() * sizeof(GLint), indices->data(), GL_STATIC_DRAW);
};
void GraphicObject::bindIndices() {
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);
};
void GraphicObject::setCamera() {
	/* Initialize transformation matrices */
	projection = glm::perspective(
		glm::radians(45.0f),
		(float)consts.SCREEN_WIDTH / (float)consts.SCREEN_HEIGHT,
		0.1f, 100.0f
	);
	view = glm::lookAt(
		consts.CAMERA_POS,
		consts.CAMERA_AT,
		consts.CAMERA_UP
	);
	// Model matrix : an identity matrix (model will be at the origin)
	model = glm::mat4(1.0f);
};
// constructors
GraphicObject::GraphicObject(vector<GLfloat>* vertices) :
	vertices(vertices),
	normals(new vector<GLfloat>()),
	indices(new vector<GLint>()),
	colors(new vector<GLfloat>()),
	objModel(NULL){
	setCamera();
	bufferVertices();

}
GraphicObject::GraphicObject(vector<GLfloat>* vertices, vector<GLfloat>* normals) :
	vertices(vertices),
	normals(normals),
	indices(new vector<GLint>()),
	colors(new vector<GLfloat>()),
	objModel(NULL){
	setCamera();
	bufferVertices();
	bufferNormals();
};

GraphicObject::GraphicObject(vector<GLfloat>* vertices, vector<GLfloat>* normals, vector<GLfloat>* colors) :
	vertices(vertices),
	normals(normals),
	colors(colors),
	indices(new vector<GLint>()),
	objModel(NULL) {
	setCamera();
	bufferVertices();
	bufferNormals();
	bufferColors();
};

GraphicObject::GraphicObject(Model* objModel) :
	objModel(objModel) {
	setCamera();
};


// geometric transformations
void GraphicObject::rotate(vec3 axis, float degree) {
	model = glm::rotate(model, degree, axis);
}
void GraphicObject::scale(float factor) {
	model = glm::scale(model, vec3(factor, factor, factor));
}
void GraphicObject::translate(vec3 v) {
	model = glm::translate(model, v);
}
// render
void GraphicObject::render(Shader* s) {
	s->use();
	s->setMat4("model", model);
	s->setMat4("projection", projection);
	s->setMat4("view", view);

	if (objModel) {
		objModel->Draw(*s);
		return;
	}

	bindVertices();
	if (normals->size() > 0)bindNormals();
	if (colors->size() > 0)bindColors();
	if (indices->size() == 0) {
		glDrawArrays(GL_TRIANGLES, 0, vertices->size());
	}
	else {
		bindIndices();
		glDrawElements(
			GL_TRIANGLES,
			indices->size(),							// Specifies the number of elements to be rendered
			GL_UNSIGNED_INT,							// Specifies the type of the values in indices.
			(void*)0);									// Specifies an offset of the first index in the array in the data store of the buffer currently bound to the GL_ELEMENT_ARRAY_BUFFER target
	}
}