#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# pylint: disable=invalid-name

"""
Biblioteca Gráfica / Graphics Library.

Desenvolvido por: <SEU NOME AQUI>
Disciplina: Computação Gráfica
Data: <DATA DE INÍCIO DA IMPLEMENTAÇÃO>
"""

import time         # Para operações com tempo
import gpu          # Simula os recursos de uma GPU
import math         # Funções matemáticas
import numpy as np  # Biblioteca do Numpy

class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800   # largura da tela
    height = 600  # altura da tela
    near = 0.01   # plano de corte próximo
    far = 1000    # plano de corte distante
    
    aspect = width/height

    top = 0
    bottom = 0
    right = 0
    left = -0
    
    VIEW = None
    
    STACK = []
    
    
    def axis_angle_to_quat(axis, angle):
    
        ax = np.asarray(axis, dtype=float)
        n = np.linalg.norm(ax)
        
        if n == 0:
            return (0.0, 0.0, 0.0, 1.0)
        
        ax = ax / n
        
        half = angle * 0.5
        
        s = np.sin(half)
        qr = np.cos(half)   
            
        qi, qj, qk = ax * s   
            
        return (qi, qj, qk, qr)
    
    def look_at(eye, at, up):
        def _norm(v): 
            return float(np.linalg.norm(v))

        def normalize(v, name, eps=1e-12):
            n = _norm(v)
            if n < eps:
                raise ValueError(f"Cannot normalize. {name}")
            return v / n

        
        w = normalize(at - eye, "at - eye")       
        u = normalize(np.cross(w, up), "up")    
        v = np.cross(u, w)                

        T = np.array([
            [1, 0, 0, -eye[0]],
            [0, 1, 0, -eye[1]],
            [0, 0, 1, -eye[2]],
            [0, 0, 0, 1]
        ])
        
        M = np.array([
            [u[0], v[0], -w[0], 0],
            [u[1], v[1], -w[1], 0],
            [u[2], v[2], -w[2], 0],
            [0, 0, 0, 1]
        ]).T

        return M @ T

    

    def rotation_matrix(qi, qj, qk, qr):
        q = np.array([qr, qi, qj, qk], dtype=float)
        
        n = np.linalg.norm(q)
        
        if n == 0:
            return np.eye(4)
        
        qr, qi, qj, qk = q / n  

        return np.array([
            [1 - 2*(qj*qj + qk*qk),   2*(qi*qj - qk*qr),   2*(qi*qk + qj*qr), 0],
            [2*(qi*qj + qk*qr),       1 - 2*(qi*qi + qk*qk), 2*(qj*qk - qi*qr), 0],
            [2*(qi*qk - qj*qr),       2*(qj*qk + qi*qr),   1 - 2*(qi*qi + qj*qj), 0],
            [0,                       0,                   0,                   1]
        ], dtype=float)
        
        
    def scale_matrix(x, y, z):
        return np.array([
            [x, 0, 0, 0],
            [0, y, 0, 0],
            [0, 0, z, 0],
            [0, 0, 0, 1]
        ])
        
    def translation_matrix(x, y, z):
        return np.array([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ])

    def perspective_matrix(far, near, right, top):
        return np.array([
            [near/right, 0, 0, 0],
            [0, near/top, 0, 0],
            [0, 0, -((far+near)/(far-near)), -(2*far*near)/(far-near)],
            [0, 0, -1, 0]
        ])

    def screen_transformation(W,H):
        return np.array([
            [W/2, 0, 0, W/2],
            [0, -H/2, 0, H/2],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    @staticmethod
    def setup(width, height, near=0.01, far=1000):
        """Definr parametros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width
        GL.height = height
        GL.near = near
        GL.far = far

    @staticmethod
    def polypoint2D(point, colors):
        """Função usada para renderizar Polypoint2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polypoint2D
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é a
        # coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista e assuma que sempre vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polypoint2D
        # você pode assumir inicialmente o desenho dos pontos com a cor emissiva (emissiveColor).
        
        color = list(map(int, list(map(lambda x: 255*x, colors["emissiveColor"]))))
        
        for p in range(0, len(point), 2):
            print([int(point[p]), int(point[p+1])])
            if 0 <= int(point[p]) < GL.width and 0 <= int(point[p+1]) < GL.height:
                gpu.GPU.draw_pixel([int(point[p]), int(point[p+1])], gpu.GPU.RGB8, color)  # altera pixel (u, v, tipo, r, g, b)
            
        
    @staticmethod
    def polyline2D(lineSegments, colors):
        """Função usada para renderizar Polyline2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polyline2D
        # Nessa função você receberá os pontos de uma linha no parâmetro lineSegments, esses
        # pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o valor da
        # coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é
        # a coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista. A quantidade mínima de pontos são 2 (4 valores), porém a
        # função pode receber mais pontos para desenhar vários segmentos. Assuma que sempre
        # vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polyline2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).

        # print("Polyline2D : lineSegments = {0}".format(lineSegments)) # imprime no terminal
        # print("Polyline2D : colors = {0}".format(colors)) # imprime no terminal as cores
        
        
        color = list(map(int, list(map(lambda x: 255*x, colors["emissiveColor"]))))
        
        print(lineSegments)
        
                
        for p in range(0, len(lineSegments)-2, 2):
            
            x1 = round(lineSegments[p])
            y1 = round(lineSegments[p+1])
            x2 = round(lineSegments[p+2])
            y2 = round(lineSegments[p+3])
            
            
            if x1 == x2:
                passo = 1 if y2 >= y1 else -1
                for u in range(round(y1), round(y2) + passo, passo):
                    if 0 <= u < GL.height and 0 <= x1 < GL.width:
                        gpu.GPU.draw_pixel([round(x1), u], gpu.GPU.RGB8, color)  # altera pixel (u, v, tipo, r, g, b)
            else:                                
                
                dx = x2 - x1
                dy = y2 - y1
                
                s = dy / dx
                
                vertical = abs(dy) >= abs(dx)
                                        
                if vertical:
                    
                    
                    if y2 > y1:
                        px1 = x1
                        px2 = x2
                        py1 = y1
                        py2 = y2
                    else:
                        px1 = x2
                        px2 = x1
                        py1 = y2
                        py2 = y1
                    
                    dx = px2 - px1
                    dy = py2 - py1
                    
                    v = px1
                    s = dx / dy
                    start_u, end_u = round(py1), round(py2)

                    for u in range(start_u, end_u+1):
                        yi = u
                        xi = round(v)

                        if 0 <= xi < GL.width and 0 <= yi < GL.height:
                            gpu.GPU.draw_pixel([xi, yi], gpu.GPU.RGB8, color)  # altera pixel (u, v, tipo, r, g, b)
                        v += s
                else:
                    
                    if x2 > x1:
                        px1 = x1
                        px2 = x2
                        py1 = y1
                        py2 = y2
                    else:
                        px1 = x2
                        px2 = x1
                        py1 = y2
                        py2 = y1
                    
                    dx = px2 - px1
                    dy = py2 - py1
                    
                    v = py1
                    s = dy / dx
                    start_u, end_u = round(px1), round(px2)
                    
                    for u in range(start_u, end_u+1):
                        xi = u
                        yi = round(v)

                        if 0 <= xi < GL.width and 0 <= yi < GL.height:
                            gpu.GPU.draw_pixel([xi, yi], gpu.GPU.RGB8, color)  # altera pixel (u, v, tipo, r, g, b)
                        v += s
                
                    
            
            
            

    @staticmethod
    def circle2D(radius, colors):
        """Função usada para renderizar Circle2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Circle2D
        # Nessa função você receberá um valor de raio e deverá desenhar o contorno de
        # um círculo.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Circle2D
        # você pode assumir o desenho das linhas com a cor emissiva (emissiveColor).

        print("Circle2D : radius = {0}".format(radius)) # imprime no terminal
        print("Circle2D : colors = {0}".format(colors)) # imprime no terminal as cores
        
        
        color = list(map(int, list(map(lambda x: 255*x, colors["emissiveColor"]))))
        
        eps = 0.5
        
        for x in range(round(radius)+1):
            for y in range(round(radius)+1):
                distance = math.sqrt(y**2 + x**2)
                
                if abs(distance - radius) <= eps:
                    gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, color)


    @staticmethod
    def triangleSet2D(vertices, colors):
        """Função usada para renderizar TriangleSet2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#TriangleSet2D
        # Nessa função você receberá os vertices de um triângulo no parâmetro vertices,
        # esses pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o
        # valor da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto.
        # Já point[2] é a coordenada x do segundo ponto e assim por diante. Assuma que a
        # quantidade de pontos é sempre multiplo de 3, ou seja, 6 valores ou 12 valores, etc.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o TriangleSet2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).
        print("TriangleSet2D : vertices = {0}".format(vertices)) # imprime no terminal
        print("TriangleSet2D : colors = {0}".format(colors)) # imprime no terminal as cores
        
        color = list(map(int, list(map(lambda x: 255*x, colors["emissiveColor"]))))
        
        
        def isInside(p1, p2, p3, x, y):
            
            def L(a, b, x, y):
                x0, y0 = a 
                x1, y1 = b
                return (y1 - y0) * (x - x0) - (x1 - x0) * (y - y0)
                
            v1 = L(p1, p2, x, y)
            v2 = L(p2, p3, x, y)
            v3 = L(p3, p1, x, y)
            
            # return (v1 > 0 and v2 > 0 and v3 > 0)
            return (v1 >= 0 and v2 >= 0 and v3 >= 0)

            
        for p in range(0, len(vertices)-5, 6):
            
            a = (vertices[p], vertices[p+1])   
            b = (vertices[p+2], vertices[p+3])   
            c = (vertices[p+4], vertices[p+5])
                                    
            xs = [a[0], b[0], c[0]]
            ys = [a[1], b[1], c[1]]
            
            x_min = math.floor(min(xs))
            x_max = math.ceil(max(xs))
            
            y_min = math.floor(min(ys))
            y_max = math.ceil(max(ys))
            
            
            
            # Ordenando Pontos
            # Fonte: https://stackoverflow.com/questions/41855695/sorting-list-of-two-dimensional-coordinates-by-clockwise-angle-using-python
            
            
            origin = [(a[0] + b[0] + c[0])/3, (a[1] + b[1] + c[1])/3]
            refvec = [0, 1]

            def clockwiseangle_and_distance(point):
                # Vector between point and the origin: v = p - o
                vector = [point[0]-origin[0], point[1]-origin[1]]
                # Length of vector: ||v||
                lenvector = math.hypot(vector[0], vector[1])
                # If length is zero there is no angle
                if lenvector == 0:
                    return -math.pi, 0
                # Normalize vector: v/||v||
                normalized = [vector[0]/lenvector, vector[1]/lenvector]
                dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     # x1*x2 + y1*y2
                diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2
                angle = math.atan2(diffprod, dotprod)
                # Negative angles represent counter-clockwise angles so we need to subtract them 
                # from 2*pi (360 degrees)
                if angle < 0:
                    return 2*math.pi+angle, lenvector
                # I return first the angle because that's the primary sorting criterium
                # but if two vectors have the same angle then the shorter distance should come first.
                return angle, lenvector
            pts = [a, b, c]
            s_pts = sorted(pts, key=clockwiseangle_and_distance)
            
            
            for x in range(x_min, x_max+1):
                for y in range(y_min, y_max+1):
                    if 0 <= x < GL.width and 0 <= y < GL.height:
                    
                        inside = isInside(s_pts[0], s_pts[1], s_pts[2], x, y)
                        
                        if inside:
                            gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, color)  # altera pixel (u, v, tipo, r, g, b)
                        
                        
            
                
                

            
            
            
        
        
        
        


    @staticmethod
    def triangleSet(point, colors):
        """Função usada para renderizar TriangleSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleSet
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e
        # assim por diante.
        # No TriangleSet os triângulos são informados individualmente, assim os três
        # primeiros pontos definem um triângulo, os três próximos pontos definem um novo
        # triângulo, e assim por diante.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, você pode assumir
        # inicialmente, para o TriangleSet, o desenho das linhas com a cor emissiva
        # (emissiveColor), conforme implementar novos materias você deverá suportar outros
        # tipos de cores.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("TriangleSet : pontos = {0}".format(point)) # imprime no terminal pontos
        print("TriangleSet : colors = {0}".format(colors)) # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        # gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel
        
        def isInside(p1, p2, p3, x, y):
            
            def L(a, b, x, y):
                x0, y0 = a 
                x1, y1 = b
                return (y1 - y0) * (x - x0) - (x1 - x0) * (y - y0)
                
            v1 = L(p1, p2, x, y)
            v2 = L(p2, p3, x, y)
            v3 = L(p3, p1, x, y)
            
            # return (v1 > 0 and v2 > 0 and v3 > 0)
            return (v1 >= 0 and v2 >= 0 and v3 >= 0)
        
        color = list(map(int, list(map(lambda x: 255*x, colors["emissiveColor"]))))
        
        for p in range(0, len(point)-5, 9):
            
            triangulo = np.array([
                [point[p], point[p+1], point[p+2], 1],  
                [point[p+3], point[p+4], point[p+5], 1],  
                [point[p+6], point[p+7], point[p+8], 1]
            ]).T
            
            
            
            triangulo_projetado = GL.perspective_matrix(GL.far, GL.near, GL.right, GL.top) @ GL.VIEW @ GL.STACK[-1] @ triangulo

            triangulo_projetado[0, :] = triangulo_projetado[0, :] / triangulo_projetado[3, :]

            triangulo_projetado[1, :] = triangulo_projetado[1, :] / triangulo_projetado[3, :]
                
            triangulo_projetado[2, :] = triangulo_projetado[2, :] / triangulo_projetado[3, :]

            triangulo_projetado[3, :] = triangulo_projetado[3, :] / triangulo_projetado[3, :]


            triangulo_projetado = GL.screen_transformation(GL.width, GL.height) @ triangulo_projetado
            
            
            a = (triangulo_projetado[0][0], triangulo_projetado[1][0])   
            b = (triangulo_projetado[0][1], triangulo_projetado[1][1])   
            c = (triangulo_projetado[0][2], triangulo_projetado[1][2])
                                    
            xs = [a[0], b[0], c[0]]
            ys = [a[1], b[1], c[1]]
            
            x_min = math.floor(min(xs))
            x_max = math.ceil(max(xs))
            
            y_min = math.floor(min(ys))
            y_max = math.ceil(max(ys))
            
            
            
            # Ordenando Pontos
            # Fonte: https://stackoverflow.com/questions/41855695/sorting-list-of-two-dimensional-coordinates-by-clockwise-angle-using-python
            
            
            origin = [(a[0] + b[0] + c[0])/3, (a[1] + b[1] + c[1])/3]
            refvec = [0, 1]

            def clockwiseangle_and_distance(point):
                # Vector between point and the origin: v = p - o
                vector = [point[0]-origin[0], point[1]-origin[1]]
                # Length of vector: ||v||
                lenvector = math.hypot(vector[0], vector[1])
                # If length is zero there is no angle
                if lenvector == 0:
                    return -math.pi, 0
                # Normalize vector: v/||v||
                normalized = [vector[0]/lenvector, vector[1]/lenvector]
                dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     # x1*x2 + y1*y2
                diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2
                angle = math.atan2(diffprod, dotprod)
                # Negative angles represent counter-clockwise angles so we need to subtract them 
                # from 2*pi (360 degrees)
                if angle < 0:
                    return 2*math.pi+angle, lenvector
                # I return first the angle because that's the primary sorting criterium
                # but if two vectors have the same angle then the shorter distance should come first.
                return angle, lenvector
            pts = [a, b, c]
            s_pts = sorted(pts, key=clockwiseangle_and_distance)
            
            
            for x in range(x_min, x_max+1):
                for y in range(y_min, y_max+1):
                    if 0 <= x < GL.width and 0 <= y < GL.height:
                    
                        inside = isInside(s_pts[0], s_pts[1], s_pts[2], x, y)
                        
                        if inside:
                            gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, color)  # altera pixel (u, v, tipo, r, g, b)
            
            
            
            
            
        

    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Função usada para renderizar (na verdade coletar os dados) de Viewpoint."""
        # Na função de viewpoint você receberá a posição, orientação e campo de visão da
        # câmera virtual. Use esses dados para poder calcular e criar a matriz de projeção
        # perspectiva para poder aplicar nos pontos dos objetos geométricos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Viewpoint : ", end='')
        print("position = {0} ".format(position), end='')
        print("orientation = {0} ".format(orientation), end='')
        print("fieldOfView = {0} ".format(fieldOfView))
        
        axis  = orientation[:3]
        angle = float(orientation[3])

        qi, qj, qk, qr = GL.axis_angle_to_quat(axis, angle)
        R4 = GL.rotation_matrix(qi, qj, qk, qr)
        R  = R4[:3, :3]                  

        fwd_cam = np.array([0.0, 0.0, -1.0])  
        up_cam  = np.array([0.0, 1.0,  0.0])  

        fwd_world = R @ fwd_cam
        up_world  = R @ up_cam

        eye = np.array(position)
        at  = eye + fwd_world
        up  = up_world

        GL.VIEW = GL.look_at(eye, at, up)
        
        GL.top = GL.near * np.tan(fieldOfView/2)
        GL.bottom = -GL.top
        GL.right = GL.top*GL.aspect
        GL.left = -GL.right

        
        

    @staticmethod
    def transform_in(translation, scale, rotation):
        
        
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_in será chamada quando se entrar em um nó X3D do tipo Transform
        # do grafo de cena. Os valores passados são a escala em um vetor [x, y, z]
        # indicando a escala em cada direção, a translação [x, y, z] nas respectivas
        # coordenadas e finalmente a rotação por [x, y, z, t] sendo definida pela rotação
        # do objeto ao redor do eixo x, y, z por t radianos, seguindo a regra da mão direita.
        # Quando se entrar em um nó transform se deverá salvar a matriz de transformação dos
        # modelos do mundo para depois potencialmente usar em outras chamadas. 
        # Quando começar a usar Transforms dentre de outros Transforms, mais a frente no curso
        # Você precisará usar alguma estrutura de dados pilha para organizar as matrizes.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Transform : ", end='')
        if translation:
            print("translation = {0} ".format(translation), end='') # imprime no terminal
        if scale:
            print("scale = {0} ".format(scale), end='') # imprime no terminal
        if rotation:
            print("rotation = {0} ".format(rotation), end='') # imprime no terminal
        print("")
        
        axis  = rotation[:3]
        angle = float(rotation[3])
        qi, qj, qk, qr = GL.axis_angle_to_quat(axis, angle)
        
        mundo = GL.translation_matrix(translation[0], translation[1], translation[2]) @ GL.rotation_matrix(qi, qj, qk, qr) @ GL.scale_matrix(scale[0], scale[1], scale[2])
        
        if len(GL.STACK)==0:
            GL.STACK.append(mundo)
        else:
            M = GL.STACK[-1]
            GL.STACK.append(mundo @ M)
            

    @staticmethod
    def transform_out():
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_out será chamada quando se sair em um nó X3D do tipo Transform do
        # grafo de cena. Não são passados valores, porém quando se sai de um nó transform se
        # deverá recuperar a matriz de transformação dos modelos do mundo da estrutura de
        # pilha implementada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Saindo de Transform")
        
        GL.STACK.pop()












    @staticmethod
    def triangleStripSet(point, stripCount, colors):
        """Função usada para renderizar TriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleStripSet
        # A função triangleStripSet é usada para desenhar tiras de triângulos interconectados,
        # você receberá as coordenadas dos pontos no parâmetro point, esses pontos são uma
        # lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x
        # do primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e assim
        # por diante. No TriangleStripSet a quantidade de vértices a serem usados é informado
        # em uma lista chamada stripCount (perceba que é uma lista). Ligue os vértices na ordem,
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("TriangleStripSet : pontos = {0} ".format(point), end='')
        for i, strip in enumerate(stripCount):
            print("strip[{0}] = {1} ".format(i, strip), end='')
        print("")
        print("TriangleStripSet : colors = {0}".format(colors)) # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def indexedTriangleStripSet(point, index, colors):
        """Função usada para renderizar IndexedTriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#IndexedTriangleStripSet
        # A função indexedTriangleStripSet é usada para desenhar tiras de triângulos
        # interconectados, você receberá as coordenadas dos pontos no parâmetro point, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor
        # da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto, point[2]
        # o valor z da coordenada z do primeiro ponto. Já point[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedTriangleStripSet uma lista informando
        # como conectar os vértices é informada em index, o valor -1 indica que a lista
        # acabou. A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("IndexedTriangleStripSet : pontos = {0}, index = {1}".format(point, index))
        print("IndexedTriangleStripSet : colors = {0}".format(colors)) # imprime as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def indexedFaceSet(coord, coordIndex, colorPerVertex, color, colorIndex,
                       texCoord, texCoordIndex, colors, current_texture):
        """Função usada para renderizar IndexedFaceSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#IndexedFaceSet
        # A função indexedFaceSet é usada para desenhar malhas de triângulos. Ela funciona de
        # forma muito simular a IndexedTriangleStripSet porém com mais recursos.
        # Você receberá as coordenadas dos pontos no parâmetro cord, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim coord[0] é o valor
        # da coordenada x do primeiro ponto, coord[1] o valor y do primeiro ponto, coord[2]
        # o valor z da coordenada z do primeiro ponto. Já coord[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedFaceSet uma lista de vértices é informada
        # em coordIndex, o valor -1 indica que a lista acabou.
        # A ordem de conexão não possui uma ordem oficial, mas em geral se o primeiro ponto com os dois
        # seguintes e depois este mesmo primeiro ponto com o terçeiro e quarto ponto. Por exemplo: numa
        # sequencia 0, 1, 2, 3, 4, -1 o primeiro triângulo será com os vértices 0, 1 e 2, depois serão
        # os vértices 0, 2 e 3, e depois 0, 3 e 4, e assim por diante, até chegar no final da lista.
        # Adicionalmente essa implementação do IndexedFace aceita cores por vértices, assim
        # se a flag colorPerVertex estiver habilitada, os vértices também possuirão cores
        # que servem para definir a cor interna dos poligonos, para isso faça um cálculo
        # baricêntrico de que cor deverá ter aquela posição. Da mesma forma se pode definir uma
        # textura para o poligono, para isso, use as coordenadas de textura e depois aplique a
        # cor da textura conforme a posição do mapeamento. Dentro da classe GPU já está
        # implementadado um método para a leitura de imagens.

        # Os prints abaixo são só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("IndexedFaceSet : ")
        if coord:
            print("\tpontos(x, y, z) = {0}, coordIndex = {1}".format(coord, coordIndex))
        print("colorPerVertex = {0}".format(colorPerVertex))
        if colorPerVertex and color and colorIndex:
            print("\tcores(r, g, b) = {0}, colorIndex = {1}".format(color, colorIndex))
        if texCoord and texCoordIndex:
            print("\tpontos(u, v) = {0}, texCoordIndex = {1}".format(texCoord, texCoordIndex))
        if current_texture:
            image = gpu.GPU.load_texture(current_texture[0])
            print("\t Matriz com image = {0}".format(image))
            print("\t Dimensões da image = {0}".format(image.shape))
        print("IndexedFaceSet : colors = {0}".format(colors))  # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Box
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Box : size = {0}".format(size)) # imprime no terminal pontos
        print("Box : colors = {0}".format(colors)) # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Sphere
        # A função sphere é usada para desenhar esferas na cena. O esfera é centrada no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da esfera que está sendo criada. Para desenha essa esfera você vai
        # precisar tesselar ela em triângulos, para isso encontre os vértices e defina
        # os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Sphere : radius = {0}".format(radius)) # imprime no terminal o raio da esfera
        print("Sphere : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def cone(bottomRadius, height, colors):
        """Função usada para renderizar Cones."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cone
        # A função cone é usada para desenhar cones na cena. O cone é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento bottomRadius especifica o
        # raio da base do cone e o argumento height especifica a altura do cone.
        # O cone é alinhado com o eixo Y local. O cone é fechado por padrão na base.
        # Para desenha esse cone você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Cone : bottomRadius = {0}".format(bottomRadius)) # imprime no terminal o raio da base do cone
        print("Cone : height = {0}".format(height)) # imprime no terminal a altura do cone
        print("Cone : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def cylinder(radius, height, colors):
        """Função usada para renderizar Cilindros."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cylinder
        # A função cylinder é usada para desenhar cilindros na cena. O cilindro é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da base do cilindro e o argumento height especifica a altura do cilindro.
        # O cilindro é alinhado com o eixo Y local. O cilindro é fechado por padrão em ambas as extremidades.
        # Para desenha esse cilindro você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Cylinder : radius = {0}".format(radius)) # imprime no terminal o raio do cilindro
        print("Cylinder : height = {0}".format(height)) # imprime no terminal a altura do cilindro
        print("Cylinder : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/navigation.html#NavigationInfo
        # O campo do headlight especifica se um navegador deve acender um luz direcional que
        # sempre aponta na direção que o usuário está olhando. Definir este campo como TRUE
        # faz com que o visualizador forneça sempre uma luz do ponto de vista do usuário.
        # A luz headlight deve ser direcional, ter intensidade = 1, cor = (1 1 1),
        # ambientIntensity = 0,0 e direção = (0 0 −1).

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("NavigationInfo : headlight = {0}".format(headlight)) # imprime no terminal

    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#DirectionalLight
        # Define uma fonte de luz direcional que ilumina ao longo de raios paralelos
        # em um determinado vetor tridimensional. Possui os campos básicos ambientIntensity,
        # cor, intensidade. O campo de direção especifica o vetor de direção da iluminação
        # que emana da fonte de luz no sistema de coordenadas local. A luz é emitida ao
        # longo de raios paralelos de uma distância infinita.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("DirectionalLight : ambientIntensity = {0}".format(ambientIntensity))
        print("DirectionalLight : color = {0}".format(color)) # imprime no terminal
        print("DirectionalLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("DirectionalLight : direction = {0}".format(direction)) # imprime no terminal

    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#PointLight
        # Fonte de luz pontual em um local 3D no sistema de coordenadas local. Uma fonte
        # de luz pontual emite luz igualmente em todas as direções; ou seja, é omnidirecional.
        # Possui os campos básicos ambientIntensity, cor, intensidade. Um nó PointLight ilumina
        # a geometria em um raio de sua localização. O campo do raio deve ser maior ou igual a
        # zero. A iluminação do nó PointLight diminui com a distância especificada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("PointLight : ambientIntensity = {0}".format(ambientIntensity))
        print("PointLight : color = {0}".format(color)) # imprime no terminal
        print("PointLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("PointLight : location = {0}".format(location)) # imprime no terminal

    @staticmethod
    def fog(visibilityRange, color):
        """Névoa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/environmentalEffects.html#Fog
        # O nó Fog fornece uma maneira de simular efeitos atmosféricos combinando objetos
        # com a cor especificada pelo campo de cores com base nas distâncias dos
        # vários objetos ao visualizador. A visibilidadeRange especifica a distância no
        # sistema de coordenadas local na qual os objetos são totalmente obscurecidos
        # pela névoa. Os objetos localizados fora de visibilityRange do visualizador são
        # desenhados com uma cor de cor constante. Objetos muito próximos do visualizador
        # são muito pouco misturados com a cor do nevoeiro.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Fog : color = {0}".format(color)) # imprime no terminal
        print("Fog : visibilityRange = {0}".format(visibilityRange))

    @staticmethod
    def timeSensor(cycleInterval, loop):
        """Gera eventos conforme o tempo passa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/time.html#TimeSensor
        # Os nós TimeSensor podem ser usados para muitas finalidades, incluindo:
        # Condução de simulações e animações contínuas; Controlar atividades periódicas;
        # iniciar eventos de ocorrência única, como um despertador;
        # Se, no final de um ciclo, o valor do loop for FALSE, a execução é encerrada.
        # Por outro lado, se o loop for TRUE no final de um ciclo, um nó dependente do
        # tempo continua a execução no próximo ciclo. O ciclo de um nó TimeSensor dura
        # cycleInterval segundos. O valor de cycleInterval deve ser maior que zero.

        # Deve retornar a fração de tempo passada em fraction_changed

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("TimeSensor : cycleInterval = {0}".format(cycleInterval)) # imprime no terminal
        print("TimeSensor : loop = {0}".format(loop))

        # Esse método já está implementado para os alunos como exemplo
        epoch = time.time()  # time in seconds since the epoch as a floating point number.
        fraction_changed = (epoch % cycleInterval) / cycleInterval

        return fraction_changed

    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#SplinePositionInterpolator
        # Interpola não linearmente entre uma lista de vetores 3D. O campo keyValue possui
        # uma lista com os valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantos vetores 3D quanto os
        # quadros-chave no key. O campo closed especifica se o interpolador deve tratar a malha
        # como fechada, com uma transições da última chave para a primeira chave. Se os keyValues
        # na primeira e na última chave não forem idênticos, o campo closed será ignorado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("SplinePositionInterpolator : set_fraction = {0}".format(set_fraction))
        print("SplinePositionInterpolator : key = {0}".format(key)) # imprime no terminal
        print("SplinePositionInterpolator : keyValue = {0}".format(keyValue))
        print("SplinePositionInterpolator : closed = {0}".format(closed))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0.0, 0.0, 0.0]
        
        return value_changed

    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Interpola entre uma lista de valores de rotação especificos."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#OrientationInterpolator
        # Interpola rotações são absolutas no espaço do objeto e, portanto, não são cumulativas.
        # Uma orientação representa a posição final de um objeto após a aplicação de uma rotação.
        # Um OrientationInterpolator interpola entre duas orientações calculando o caminho mais
        # curto na esfera unitária entre as duas orientações. A interpolação é linear em
        # comprimento de arco ao longo deste caminho. Os resultados são indefinidos se as duas
        # orientações forem diagonalmente opostas. O campo keyValue possui uma lista com os
        # valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantas rotações 3D quanto os
        # quadros-chave no key.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("OrientationInterpolator : set_fraction = {0}".format(set_fraction))
        print("OrientationInterpolator : key = {0}".format(key)) # imprime no terminal
        print("OrientationInterpolator : keyValue = {0}".format(keyValue))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0, 0, 1, 0]

        return value_changed

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""
