from motion_detection import dense_optical_flow, background_subtraction_mog2, running_average, sparse_optical_flow
from multiprocessing import Process

if __name__ == '__main__':
    path = ("cars.mp4")

    process1 = Process(target=dense_optical_flow, args=(path,))
    process2 = Process(target=background_subtraction_mog2, args=(path,))
    process3 = Process(target=running_average, args=(path,))
    process4 = Process(target=sparse_optical_flow, args=(path,))


    process1.start()
    process2.start()
    process3.start()
    process4.start()

    process1.join()
    process2.join()
    process3.join()
    process4.join()


# running_average(path)
# background_subtraction_mog2(path)
# dense_optical_flow(path)
# sparse_optical_flow(path)