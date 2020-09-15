#include <iostream>
#include <vector>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/ext.hpp>

using namespace std;
using namespace glm;
typedef unsigned char byte;


class ObjLoader {
	string filepath;

public:
	ObjLoader(string filepath) :filepath(filepath) {}
};
class BVoxLoader {
	byte* voxels = 0;
	vector<GLfloat>* vertices;
	vector<GLfloat>* normals;
	vector<GLint>* indices;

	void pushVertex(vector<GLfloat>* vertices, vec3 v) {
		vertices->push_back(v.x);
		vertices->push_back(v.y);
		vertices->push_back(v.z);
	}

	int get_index(int x, int y, int z)
	{
		int index = x * (width * height) + z * width + y;
		return index;
	}

	void initializeBuffers() {
		vertices = new vector<GLfloat>();
		normals = new vector<GLfloat>();

		float deltha = 1.0 / width;
		for (int i = 0; i < width; i++)
			for (int j = 0; j < height; j++)
				for (int k = 0; k < depth; k++) {
					int index = get_index(i, j, k);
					if (voxels[index]) {

						vec3 a = vec3(i * deltha,j * deltha,k * deltha);
						vec3 b = vec3((i + 1) * deltha,j * deltha,k * deltha);
						vec3 c = vec3((i + 1) * deltha,j * deltha,(k + 1) * deltha);
						vec3 d = vec3(i * deltha,j * deltha,(k + 1) * deltha);
						vec3 a1 = vec3(i * deltha, (j+1) * deltha, k * deltha);
						vec3 b1 = vec3((i + 1) * deltha, (j+1) * deltha, k * deltha);
						vec3 c1 = vec3((i + 1) * deltha, (j+1) * deltha, (k + 1) * deltha);
						vec3 d1 = vec3(i * deltha, (j+1) * deltha, (k + 1) * deltha);

						vec3 up = vec3(0, 1, 0);
						vec3 down = vec3(0, -1, 0);
						vec3 front = vec3(0, 0, 1);
						vec3 back = vec3(0, 0, -1);
						vec3 left = vec3(-1, 0, 0);
						vec3 right = vec3(1, 0, 0);

						// A B C D  (A B C  A C D )
						for (int ii = 0; ii < 6;ii++)
							pushVertex(normals, down);
						pushVertex(vertices, a);
						pushVertex(vertices, b);
						pushVertex(vertices, c);
						pushVertex(vertices, a);
						pushVertex(vertices, c);
						pushVertex(vertices, d);
						// A'B'C'D' (A'B'C' A'C'D')
						for (int ii = 0; ii < 6; ii++)
							pushVertex(normals, up);
						pushVertex(vertices, a1);
						pushVertex(vertices, b1);
						pushVertex(vertices, c1);
						pushVertex(vertices, a1);
						pushVertex(vertices, c1);
						pushVertex(vertices, d1);
						// D C D'C' (D C C' D C'D')
						for (int ii = 0; ii < 6; ii++)
							pushVertex(normals, front);
						pushVertex(vertices, d);
						pushVertex(vertices, c);
						pushVertex(vertices, c1);
						pushVertex(vertices, d);
						pushVertex(vertices, c1);
						pushVertex(vertices, d1);
						// A B B'A' (A B B' A B'A')
						for (int ii = 0; ii < 6; ii++)
							pushVertex(normals, back);
						pushVertex(vertices, a);
						pushVertex(vertices, b);
						pushVertex(vertices, b1);
						pushVertex(vertices, a);
						pushVertex(vertices, b1);
						pushVertex(vertices, a1);
						// C B B'C' (C B B' C B'C')
						for (int ii = 0; ii < 6; ii++)
							pushVertex(normals, right);
						pushVertex(vertices, c);
						pushVertex(vertices, b);
						pushVertex(vertices, b1);
						pushVertex(vertices, c);
						pushVertex(vertices, b1);
						pushVertex(vertices, c1);
						// D A A'D' (D A A' D A'D')
						for (int ii = 0; ii < 6; ii++)
							pushVertex(normals, left);
						pushVertex(vertices, d);
						pushVertex(vertices, a);
						pushVertex(vertices, a1);
						pushVertex(vertices, d);
						pushVertex(vertices, a1);
						pushVertex(vertices, d1);
					}
				}
	}

public:

	float tx, ty, tz;
	float scale;
	int version;
	int depth, height, width;
	int size;

	BVoxLoader(string filepath) {
		ifstream* input = new ifstream(filepath.c_str(), ios::in | ios::binary);

		//
		// read header
		//
		string line;
		*input >> line;  // #binvox
		if (line.compare("#binvox") != 0) {
			cout << "Error: first line reads [" << line << "] instead of [#binvox]" << endl;
			delete input;
			return;
		}
		*input >> version;
		cout << "reading binvox version " << version << endl;

		depth = -1;
		int done = 0;
		while (input->good() && !done) {
			*input >> line;
			if (line.compare("data") == 0) done = 1;
			else if (line.compare("dim") == 0) {
				*input >> depth >> height >> width;
			}
			else if (line.compare("translate") == 0) {
				*input >> tx >> ty >> tz;
			}
			else if (line.compare("scale") == 0) {
				*input >> scale;
			}
			else {
				cout << "  unrecognized keyword [" << line << "], skipping" << endl;
				char c;
				do {  // skip until end of line
					c = input->get();
				} while (input->good() && (c != '\n'));

			}
		}
		if (!done) {
			cout << "  error reading header" << endl;
			return;
		}
		if (depth == -1) {
			cout << "  missing dimensions in header" << endl;
			return;
		}

		if (width != height || width != depth) {
			cout << " width height and depth should be the same";
			return;
		}

		size = width * height * depth;
		voxels = new byte[size];
		if (!voxels) {
			cout << "  error allocating memory" << endl;
			return;
		}

		//
		// read voxel data
		//
		byte value;
		byte count;
		int index = 0;
		int end_index = 0;
		int nr_voxels = 0;

		input->unsetf(ios::skipws);  // need to read every byte now (!)
		*input >> value;  // read the linefeed char

		while ((end_index < size) && input->good()) {
			*input >> value >> count;

			if (input->good()) {
				end_index = index + count;
				if (end_index > size) return;
				for (int i = index; i < end_index; i++) voxels[i] = value;

				if (value) nr_voxels += count;
				index = end_index;
			}  // if file still ok

		}  // while

		input->close();
		cout << "  read " << nr_voxels << " voxels" << endl;
		initializeBuffers();

	};
	vector<GLfloat>* getVertices() { return vertices; }
	vector<GLfloat>* getNormals() { return normals; }
};


