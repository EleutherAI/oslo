import os

import torch


def main():
    file_names = os.listdir("tmp2/")
    no_pp_names = sorted([fn for fn in file_names if "no_pp" in fn])

    print(len(no_pp_names))

    diff_cnt = 0
    for no_pp_name in no_pp_names:
        pp_name = no_pp_name.replace("no_pp", "pp")

        pp_path = os.path.join("tmp2", pp_name)
        no_pp_path = os.path.join("tmp2", no_pp_name)

        pp_data = torch.load(pp_path, map_location="cpu")
        no_pp_data = torch.load(no_pp_path, map_location="cpu")

        if not torch.allclose(pp_data, no_pp_data):
            # print(torch.abs(pp_data - no_pp_data))
            # print(pp_name)

            diff_cnt += 1

            # break

    print(diff_cnt)


if __name__ == "__main__":
    main()
