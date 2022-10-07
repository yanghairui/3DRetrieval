import numpy as np
import re
import moderngl
import numpy as np

from pyrr import Matrix44
from PIL import Image


def load_off(file_name):
    try:
        f = open(file_name, 'r')
    except Exception as e:
        print('load off file failed!')
        return None
    lines = f.readlines()
    if lines[0] == 'OFF\n':
        start_line = 2
    elif re.match('^OFF', lines[0]) is not None:
        start_line = 1
    else:
        raise IOError("NOT OFF FILE")
    split_strings = [line.rstrip().split(' ') for line in lines]
    vertices = []
    out_vertices = []
    state = 0
    for i in range(start_line, len(split_strings)):
        arr = split_strings[i]
        if len(arr) == 3 and state == 0:
            vertex = [float(v) for v in arr]
            vertex = np.array(vertex)
            vertices.append(vertex)
        elif len(arr) == 4:
            state = 1
            c, v1, v2, v3 = arr
            assert c == '3'
            v1, v2, v3 = int(v1), int(v2), int(v3)
            out_vertices.append([vertices[v1], vertices[v2], vertices[v3]])
        else:
            raise IOError('wrong file format')
    f.close()
    out_vertices = np.array(out_vertices)
    # to avoid overflow
    out_vertices /= np.max(np.abs(out_vertices))
    l10 = out_vertices[:, 0, :] - out_vertices[:, 1, :]
    l02 = out_vertices[:, 2, :] - out_vertices[:, 0, :]
    normals = np.cross(l10, l02)
    centroids = out_vertices.mean(1)
    weights = np.expand_dims(np.linalg.norm(normals, axis=1), 0)
    centroid = weights.dot(centroids) / weights.sum()
    out_vertices = out_vertices.reshape(-1, 3)
    out_vertices -= centroid
    max_length = np.max(np.linalg.norm(out_vertices.reshape(-1, 3), axis=1))
    out_vertices /= max_length

    return out_vertices, np.expand_dims(normals, 1).repeat(3, axis=1).reshape(-1, 3)


dodecahedron_polar_pos = [[0.78539816, 0.61547971],
                          [0.78539816, -0.61547971],
                          [-0.78539816, 0.61547971],
                          [-0.78539816, -0.61547971],
                          [-0.78539816, 0.61547971],
                          [-0.78539816, -0.61547971],
                          [0.78539816, 0.61547971],
                          [0.78539816, -0.61547971],
                          [1.57079633, 1.2059325],
                          [1.57079633, -1.2059325],
                          [-1.57079633, 1.2059325],
                          [-1.57079633, -1.2059325],
                          [0., 0.36486383],
                          [0., -0.36486383],
                          [-0., 0.36486383],
                          [-0., -0.36486383],
                          [1.2059325, 0.],
                          [-1.2059325, 0.],
                          [-1.2059325, 0.],
                          [1.2059325, 0.]]


class Render(object):
    def __init__(self, ctx=None):
        if ctx is None:
            self.ctx = moderngl.create_standalone_context()
        else:
            self.ctx = ctx
        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 Mvp;
                in vec3 in_vert;
                in vec3 in_norm;
                out vec3 v_vert;
                out vec3 v_norm;
                void main() {
                    v_vert =  in_vert;
                    v_norm =  in_norm;
                    gl_Position = Mvp*vec4(v_vert, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                uniform vec3 Light;
                in vec3 v_vert;
                in vec3 v_norm;
                out vec4 f_color;
                void main() {
                    vec3 light = Light - v_vert;
                    float d_light = length(light);
                    float lum = abs(dot(normalize(light), normalize(v_norm)));
                    lum = clamp(45.0/(d_light*(d_light+0.02)) * lum, 0.0,1.0)* 0.6 +0.3;
                    f_color = vec4(lum * vec3(1.0, 1.0, 1.0), 0.0);
                }
            ''',
        )

        self.vbo_vertices = None
        self.vbo_normals = None
        self.vao = None
        self.fbo = None
        # uniform variables
        self.light = self.prog['Light']
        self.mvp = self.prog['Mvp']

    def setViewport(self, viewport):
        self.ctx.viewport = viewport

    def load_model(self, vertices, normals):
        vertices = vertices.flatten()
        normals = normals.flatten()
        if self.vbo_vertices is not None:
            self.vbo_vertices.release()
        if self.vbo_normals is not None:
            self.vbo_normals.release()
        self.vbo_vertices = self.ctx.buffer(vertices.astype(np.float32).tobytes())
        self.vbo_normals = self.ctx.buffer(normals.astype(np.float32).tobytes())
        if self.vao is not None:
            self.vao.release()
        self.vao = self.ctx.vertex_array(self.prog, [
            (self.vbo_vertices, '3f', 'in_vert'),
            (self.vbo_normals, '3f', 'in_norm'),
        ])

    def images(self, off_file, num_views=12, use_dodecahedron_views=False):
        model = load_off(off_file)
        self.load_model(*model)
        return self.render_to_images(output_views=num_views, use_dodecahedron_views=use_dodecahedron_views)

    def render_frame(self, theta, phi=30 / 180 * np.pi):
        self.ctx.clear(1.0, 1.0, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        camera_r = 3.88  # >= 1 / sin(pi/12)
        light_r = 6.5
        cos_theta, sin_theta, cos_phi, sin_phi = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
        camera_pos = (cos_theta * cos_phi * camera_r, sin_theta * cos_phi * camera_r, sin_phi * camera_r)
        self.light.value = (cos_theta * cos_phi * light_r, sin_theta * cos_phi * light_r, sin_phi * light_r)

        proj = Matrix44.perspective_projection(30.0, 1, 0.1, 1000.0)
        lookat = Matrix44.look_at(
            camera_pos,
            (0.0, 0.0, 0.0),  # look at origin
            (0.0, 0.0, 1.0),  # camera orientation
        )
        self.mvp.write((proj * lookat).astype('f4').tobytes())
        self.vao.render()

    def render_to_images(self, output_views=12, use_dodecahedron_views=False):
        """
        Render the model to `PIL` images
        :param output_views: render views count
        :param use_dodecahedron_views: use regular dodecahedron (20 vertices), output_views is `ignored` if True
        :return: a list of images
        """

        if self.fbo is None:
            self.fbo = self.ctx.simple_framebuffer((256, 256))
        self.fbo.use()
        images = []
        if use_dodecahedron_views:
            for theta, phi in dodecahedron_polar_pos:
                self.render_frame(theta, phi)
                image = Image.frombytes('RGB', self.fbo.size, self.fbo.read(), 'raw', 'RGB', 0, -1)
                images.append(image)
        else:
            delta_theta = 2 * np.pi / output_views
            for i in range(output_views):
                angle = delta_theta * i
                self.render_frame(angle)
                image = Image.frombytes('RGB', self.fbo.size, self.fbo.read(), 'raw', 'RGB', 0, -1)
                images.append(image)
        self.fbo.clear()
        return images

    def render_and_save(self, off_file, output_dir, output_views=12, use_dodecahedron_views=False):
        self.load_model(*ol.load_off(off_file))
        images = self.render_to_images(output_views, use_dodecahedron_views=use_dodecahedron_views)
        self._save_images(images, off_file, output_dir)

    # def _save_images_in_parallel(self, images, off_file, output_dir):
    #     import threading as th
    #     th.Thread(target=Render._save_images(images, off_file, output_dir)).start()

    @staticmethod
    def _save_images(images, off_file, output_dir):
        for i, image in enumerate(images):
            image = image.resize((224, 224), Image.BICUBIC)
            image.save("%s/%s_%03d.jpg" % (output_dir, off_file.split('.')[0].split('/')[-1], i))

RENDER = Render()