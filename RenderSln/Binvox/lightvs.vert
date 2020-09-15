#version 460 core

// vertex attributes
layout(location = 0) in vec3 pos;

// transformation
uniform mat4 model;
uniform mat4 projection;
uniform mat4 view;

// light
uniform vec3 v_light_color;
out vec3 light_color;

void main(){
	light_color = v_light_color;
	gl_Position = projection * view * model * vec4(pos.x,pos.y,pos.z,1);
}