import tensorflow as tf


def main():
    model = tf.keras.models.load_model('../models/002_sim_M2_2D/tfModel')

    print("Some tests")
    test_u = [[0.5, 0.5]]
    res = model(test_u)

    print(res)

    return 0


if __name__ == '__main__':
    main()
