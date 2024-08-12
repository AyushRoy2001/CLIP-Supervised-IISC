import numpy as np
import pandas as pd


def worst_perf_images(df, n=100):      ## Return a list of image names which perform the worst in terms of absolute difference between 'mos' and 'pred'
    df["difference"] = df["mos"]-df["pred"]
    df["difference"] = df["difference"].apply(lambda x: -x if x<0 else x)
    df = df.sort_values(by=['difference'], ascending=False)
    print(df["difference"].head(5))
    image_loc = np.asarray(df["im_loc"])
    return np.ndarray.tolist(image_loc[:n])


def main():
    df = pd.read_csv('D:/IISC interview/Interview/contrique_livefb_test.csv', index_col=[0])
    print(df)
    print(worst_perf_images(df))


if __name__ == '__main__':
    main()
