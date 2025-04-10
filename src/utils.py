import os
import torch as th
import json 
from datetime import datetime


def save_model(model, losses, env):

    #Create model directory
    base_dir = os.path.join("outputs", "savedmodels")
    today = datetime.now().strftime("%Y-%m-%d")
    date_dir = os.path.join(base_dir, today)
    os.makedirs(date_dir, exist_ok=True)
    model_dir = os.path.join(date_dir, model.name)
    os.makedirs(model_dir, exist_ok=True)

    weight_file = os.path.join(model_dir, "weights")
    log_file = os.path.join(model_dir, "log.json")
    cfg_file = os.path.join(model_dir, "cfg.json")

    #Saving model weights, training history and environment configuration dictionary
    th.save(model.state_dict(), weight_file)
    with open(log_file, 'w') as file:
        json.dump(losses, file)
    cfg = env.get_save_config()
    with open(cfg_file, 'w') as file:
        json.dump(cfg, file)

    print(f"Done saving model's weights, training history and env in {model_dir}")

def load_environment(cfg_file, verbose=False):

    with open(cfg_file, 'r') as file:
        cfg = json.load(file)

    if verbose:
        for k1, v1 in cfg.items():
            if isinstance(v1, dict):
                print(k1 + ":")
                for k2, v2 in v1.items():
                    if type(v2) is dict:
                        print("\t\t" + k2 + ":")
                        for k3, v3 in v2.items():
                            print("\t\t\t\t" + k3 + ": ", v3)
                    else:
                        print("\t\t" + k2 + ": ", v2)
            else:
                print(k1 + ": ", v1)

    return cfg

def load_model(env, model, weight_file):

    return model.load_state_dict(th.load(weight_file))
