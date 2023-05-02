import numpy as np


def generate_graph(w, h, min_nodes=6, max_nodes=32, min_connections_per_node=1, max_connection_per_node=16,
                   min_depth=0.5, max_depth=2, min_permeability=0.2, max_permeability=1.2,
                   min_height=0.2, max_height=2):
    margin_x = w // 10
    margin_y = h // 10
    nodes = [(np.random.randint(margin_x, w-margin_x),
              np.random.randint(margin_y, h-margin_y))
             for _ in range(np.random.randint(min_nodes, max_nodes+1))]

    connection_counts = np.zeros(len(nodes), dtype=int)
    connected_to = [set() for _ in nodes]
    connections = []

    for i in range(len(nodes)):
        conns = np.random.randint(min_connections_per_node, max_connection_per_node)
        conns = min(conns, len(nodes) - 1)
        if connection_counts[i] >= conns:
            continue
        x, y = nodes[i]
        dsts = np.array([(_x - x) ** 2 + (_y - y) ** 2 for _x, _y in nodes])
        dsts[i] = np.max(dsts) * 2
        probas = np.e**-(4 * dsts / (w**2 + h**2))
        probas = probas / np.sum(probas)

        for _ in range(conns - connection_counts[i]):
            j = np.random.choice(len(nodes), p=probas)
            while j == i or j in connected_to[i]:
                j = np.random.choice(len(nodes), p=probas)
            connections.append((i, j))
            connected_to[i].add(j)
            connected_to[j].add(i)
            connection_counts[i] += 1
            connection_counts[j] += 1

    map_mask = np.zeros((h, w))
    intersections_mask = np.zeros((h, w))
    intersections = set()
    pts = set()
    for u, v in connections:
        x, y = nodes[u]
        x_t, y_t = nodes[v]

        while x != x_t or y != y_t:
            if map_mask[y, x] != 0:
                intersections.add((x, y))
                intersections_mask[y, x] += 1
            map_mask[y, x] = 1
            pts.add((x, y))

            dy = y_t - y
            dx = x_t - x

            if np.random.rand() < abs(dx) / (abs(dx) + abs(dy)):
                x += np.sign(dx) if x - np.sign(dx) < 0 or x - np.sign(dx) >= w or np.random.rand() > 0.1 else -np.sign(dx)
            else:
                y += np.sign(dy) if y - np.sign(dy) < 0 or y - np.sign(dy) >= h or np.random.rand() > 0.1 else -np.sign(dy)

        if map_mask[y, x] != 0:
            intersections.add((x, y))
            intersections_mask[y, x] += 1
        map_mask[y, x] = 1
        pts.add((x, y))

    centers = np.array(list(intersections))
    nodes = set(nodes)
    is_node = np.array([[[(pt in nodes) for pt in list(intersections)]]])
    pts = np.array(list(pts))

    widths = np.clip((1.5 * (w + h) / len(centers)) * ((1.3 + is_node * 0.8) + (1.2 + 1.2 * is_node) * np.random.rand(1, 1, len(centers))), 0.5, None)
    depths = min_depth + np.random.random(len(centers)) * (max_depth - min_depth)
    permeabilities = min_permeability + np.random.random(len(centers)) * (max_permeability - min_permeability)
    porosities = 0.1 + 0.9 * np.random.random(len(centers))**0.5
    heights = min_height + (max_height - min_height) * np.random.random(len(centers))**1.25

    column = np.linspace(0, w-1, w).reshape((1, -1, 1)).repeat(h, axis=0).repeat(len(centers), axis=-1)
    row = np.linspace(0, h-1, h).reshape((-1, 1, 1)).repeat(w, axis=1).repeat(len(centers), axis=-1)
    dst = np.stack([column, row], axis=-1)

    ctr = np.array(centers).reshape((1, 1, -1, 2)).repeat(h, axis=0).repeat(w, axis=1)
    dst = ((ctr - dst)**2).sum(-1)**0.5

    column = np.linspace(0, w-1, w).reshape((1, -1, 1)).repeat(h, axis=0).repeat(len(pts), axis=-1)
    row = np.linspace(0, h-1, h).reshape((-1, 1, 1)).repeat(w, axis=1).repeat(len(pts), axis=-1)
    pts_dst = np.stack([column, row], axis=-1)
    pts_dst = ((pts - pts_dst)**2).sum(-1)**0.5

    decay_coef = (np.clip(dst / widths, 0., None).min(axis=-1) * np.clip(pts_dst, 0., None).min(axis=-1))
    decay = 1.7 ** -np.clip(decay_coef - 0.2, 0., None)

    coefs = 1.9 ** -(dst / widths)
    coefs = coefs / np.sum(coefs, axis=-1, keepdims=True)

    depth_coefs = 2.2 ** -(dst / widths)
    depth_coefs = depth_coefs / np.sum(depth_coefs, axis=-1, keepdims=True)

    permeability_coefs = 2.1 ** -(dst / widths)
    permeability_coefs = permeability_coefs / np.sum(permeability_coefs, axis=-1, keepdims=True)

    depth_map = (depths * depth_coefs).sum(axis=-1)
    permeability_map = (permeabilities * permeability_coefs).sum(axis=-1)
    porosity_map = (porosities * coefs).sum(axis=-1) * decay
    height_map = (heights * coefs).sum(axis=-1) * decay

    return np.stack([permeability_map, depth_map, height_map, porosity_map], axis=-1)
