import math


def rotate_vector(vector, angle):
    x, y = vector
    cos_theta = math.cos(angle)
    sin_theta = math.sin(angle)
    x_new = x * cos_theta - y * sin_theta
    y_new = x * sin_theta + y * cos_theta
    return [x_new, y_new]


vec1 = [2.0, 0.0]
vec_45 = rotate_vector(vec1, angle=math.pi / 4)
vec_m45 = rotate_vector(vec1, angle=-math.pi / 4)

print(vec_45)
print(vec_m45)
