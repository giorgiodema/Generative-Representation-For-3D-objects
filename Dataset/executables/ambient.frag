#version 330 core
out vec4 FragColor;

in vec3 light_color;
in vec3 light_pos;
in float ambient_strenght;

in vec3 pos;  
in vec3 norm;
in vec3 color;

uniform sampler2D texture_diffuse1;

void main()
{    
	// ambient component
	vec3 ambient = light_color * ambient_strenght;
	// diffuse component
	vec3 n = normalize(norm);
	vec3 light_dir = normalize(light_pos - pos);
	float diff = max(dot(n, light_dir), 0.0);
	vec3 diffuse = diff * light_color;
	// result
	vec3 aux = (ambient + diffuse);
	vec4 result = vec4(aux.x,aux.y,aux.z,1.0);

    FragColor = result * vec4(color.x,color.y,color.z,1.0);
}