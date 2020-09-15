#version 460 core

layout(location = 0) in vec3 v_pos;
layout(location = 1) in vec3 v_norm;
uniform mat4 model;
uniform mat4 projection;
uniform mat4 view;

uniform vec3 v_light_color;
uniform vec3 v_voxel_color;
uniform vec3 v_light_pos;
uniform float v_ambient_strenght;
out vec3 light_color;
out vec3 voxel_color;
out vec3 light_pos;
out float ambient_strenght;

out vec3 pos;  
out vec3 norm;

void main(){
	light_color = v_light_color;
	voxel_color = v_voxel_color;
	ambient_strenght = v_ambient_strenght;
	light_pos = v_light_pos;
	
	norm = vec3(model*vec4(v_norm,1.0));
	pos = vec3(model * vec4(v_pos, 1.0));

	gl_Position = projection * view * model * vec4(v_pos.x,v_pos.y,v_pos.z,1);
}