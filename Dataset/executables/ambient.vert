#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec3 aColor;

uniform vec3 v_light_color;
uniform vec3 v_light_pos;
uniform float v_ambient_strenght;
out vec3 light_color;
out vec3 light_pos;
out float ambient_strenght;

out vec3 color;

out vec3 pos;  
out vec3 norm;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{

	light_color = v_light_color;
	ambient_strenght = v_ambient_strenght;
	light_pos = v_light_pos;

	norm = vec3(model*vec4(aNormal,0));
	pos = vec3(model * vec4(aPos, 1.0));

    color = aColor;    
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}