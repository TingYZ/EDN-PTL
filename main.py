from src.train import train
from src.test import pred
import numpy as np

if __name__ == "__main__":

    # train model
    # train()

    # test
    rmse, rmse_ave, mr = pred("TestSet")

    print("rmse:", np.around(rmse, 2))
    print("rmse_avg:", round(rmse_ave, 2))
    print("mr%:", round(mr * 100, 2))
