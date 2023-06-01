import os
import glob

import torch


main_dir = "tmp2"

compair_dirs = ["tmp3"]


def main():
    file_names = os.listdir(f"{main_dir}/")

    diff_names = set()
    same_names = set()
    for name in file_names:
        left_path = os.path.join(main_dir, name)
        left = torch.load(left_path, map_location="cpu")

        for rd in compair_dirs:
            right_path = left_path.replace(main_dir, rd)
            right = torch.load(right_path, map_location="cpu")

            if not torch.allclose(left, right):
                diff_names.add(name)
            else:
                same_names.add(name)

    print("Names with difference gradient: ")
    for dn in diff_names:
        print(dn)

    print(f"{len(diff_names)} / {len(file_names)}")

    print("Names with same gradient: ")
    for sn in same_names:
        print(sn)

    print(f"{len(same_names)} / {len(file_names)}")


if __name__ == "__main__":
    main()
