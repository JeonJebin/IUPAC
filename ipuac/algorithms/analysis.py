def find_connected_compounds(image, vertex, threshold):
    th = len(vertex)
    preprocessed_image = image.preprocess_for_body_analysis(threshold)

    edge = []
    for i in range(0, len(vertex) - 1):
        for j in range(i + 1, len(vertex)):
            count = 0
            for k in range(1, th):
                temp = [int(((th - k) * vertex[i][1] + k * vertex[j][1]) / th),
                        int(((th - k) * vertex[i][0] + k * vertex[j][0]) / th)]
                if preprocessed_image[temp[0], temp[1]] < 55:
                    count += 1

            if count >= int(th * 0.8):
                edge.append([i, j])

    # template.show_connected_compounds(image, preprocessed_image, vertex, edge, threshold)

    return edge


