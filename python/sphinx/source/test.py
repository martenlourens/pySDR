import pathlib

if __name__ == "__main__":
    print(pathlib.Path(__file__).parents[2].resolve().as_posix())