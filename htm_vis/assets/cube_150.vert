#version 150 core
in vec3 a_pos;
in uint a_type;
out vec3 v_clr;
uniform mat4 u_model_view_proj;

void main() {
    const float scale = 0.5;
    const vec3 A = scale * vec3(-1, -1, -1);// left bottom front
    const vec3 B = scale * vec3( 1, -1, -1);// right bottom front
    const vec3 C = scale * vec3( 1, -1,  1);// right bottom back
    const vec3 D = scale * vec3(-1, -1,  1);// left bottom back
    const vec3 E = scale * vec3(-1,  1, -1);// left top front
    const vec3 F = scale * vec3( 1,  1, -1);// right top front
    const vec3 G = scale * vec3( 1,  1,  1);// right top back
    const vec3 H = scale * vec3(-1,  1,  1);// left top back

    const vec3[6*6] direction_per_vertex = vec3[6*6](
        // XPlus ortientation = block's right face
        G, B, F, B, G, C,
        // XMinus ortientation = block's left face
        A, D, H, A, H, E,
        // YPlus ortientation = block's top face
        G, F, E, G, E, H,
        // YMinus ortientation = block's bottom face
        C, A, B, C, D, A,
        // ZPlus ortientation = block's back face
        H, D, C, G, H, C,
        // ZMinus ortientation = block's front face
        F, B, A, F, A, E
    );

    const vec3[6] COLOR_TYPES = vec3[6](
        vec3(0.9,0.9,0.9), // INACTIVE_COLOR
        vec3(0.5,1.,0.5), // ACTIVE_COLOR
        vec3(0.5,0.5,0.5), // SELECTED_INACTIVE_COLOR
        vec3(0.2,0.5,0.2), // SELECTED_ACTIVE_COLOR
        vec3(0.5,0.5,1), // POTENTIAL_INACTIVE_COLOR
        vec3(0.2,0.2,0.5) // POTENTIAL_ACTIVE_COLOR
    );

    vec3 vertex = direction_per_vertex[gl_VertexID];
    v_clr = COLOR_TYPES[a_type] + vertex*0.2;
    gl_Position = u_model_view_proj * vec4(a_pos + vertex, 1.0);
}