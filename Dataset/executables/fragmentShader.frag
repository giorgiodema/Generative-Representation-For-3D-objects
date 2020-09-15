#version 460 core

out vec4 FragColor;

in vec3 light_color;
in vec3 voxel_color;
in vec3 light_pos;
in float ambient_strenght;

in vec3 pos;  
in vec3 norm;

void main(){

	// ambient component
	vec3 ambient = light_color * ambient_strenght;
	// diffuse component
	vec3 n = normalize(norm);
	vec3 light_dir = normalize(light_pos - pos);
	float diff = max(dot(n, light_dir), 0.0);
	vec3 diffuse = diff * light_color;
	// result
	vec3 result = (ambient + diffuse) * voxel_color;

	FragColor = vec4(result, 1.0);
}