def read_file(file_path):
    with open(file_path, 'r') as f:
        txt = f.read()
  
    vertices_and_classes = txt.split('\n')

    v_list = []
    c_list = []

    for v in vertices_and_classes:
        vertices, classes = v.split()
        vertices = list(map(int, vertices.split(',')))
        classes = list(map(int, classes.split(',')))  

        vertices = [vertices[i:i+2] for i in range(0, len(vertices), 2)]
        v_list.append(vertices)
        c_list.append(classes)
    
    return v_list, c_list