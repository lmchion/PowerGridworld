import os,pickle,json

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(THIS_DIR, "data/state_space_model.p"), "rb") as f:
        models = pickle.load(f)

with open(os.path.join(THIS_DIR, "data/state_space_model.json"), "w") as outfile:
    outfile.write(str(models))

if __name__ == '__main__':
        pass