import os
import glob

import torch
import einops


save_dir = "tmp_tp"


def main():
    file_names = os.listdir(f"{save_dir}/")
    no_tp_names = sorted([fn for fn in file_names if "no_tp" in fn and "output_" in fn])

    print(len(no_tp_names))

    diff_cnt = 0
    diff_names = []
    same_names = []
    for no_tp_name in no_tp_names:
        tp_name_template = no_tp_name.split("no_tp")[0]
        tp_name_template = tp_name_template + "tp_*.pkl"
        tp_paths = glob.glob(os.path.join(f"{save_dir}", tp_name_template))
        tp_paths = sorted(tp_paths)

        print()
        print(tp_name_template)
        print(tp_paths)

        no_tp_path = os.path.join(f"{save_dir}", no_tp_name)
        no_tp_data = torch.load(no_tp_path, map_location="cpu")

        tp_data = []
        for path in tp_paths:
            data = torch.load(path, map_location="cpu")
            tp_data.append(data)

        # TODO
        if not torch.is_tensor(no_tp_data):
            print(f"pass {no_tp_path}")
            continue

        print(f"{tp_data[0].shape=}, {no_tp_data.shape=}")

        if any([x in no_tp_name for x in [".attn_dropout"]]):
            tp_data = torch.cat(tp_data, 1)

        elif any([x in no_tp_name for x in [".mlp.act", ".mlp.c_fc"]]):
            tp_data = torch.cat(tp_data, -1)

        # qkv reshape
        elif any([x in no_tp_name for x in [".c_attn"]]):
            for i in range(len(tp_data)):
                tp_data[i] = einops.rearrange(tp_data[i], "b t (n d) -> b t n d", n=3)
            tp_data = torch.stack(tp_data, 0)
            tp_data = einops.rearrange(tp_data, "m b t n d -> b t (n m d)")

        # no split
        else:
            tp_data = tp_data[0]

        print(f"{tp_data.shape=}, {no_tp_data.shape=}")

        if not torch.allclose(tp_data, no_tp_data):
            print(f"   >>> diff: {torch.sum(torch.abs(tp_data - no_tp_data))}")
            # print(pp_name)

            diff_cnt += 1
            diff_names.append(no_tp_name)

            # break
        else:
            same_names.append(no_tp_name)

    print("Names with different output: ")
    for dn in diff_names:
        print(dn.replace("_no_pp.pkl", ""))

    print(f"{diff_cnt} / {len(no_tp_names)}")

    print("Names with same output: ")
    for sn in same_names:
        print(sn.replace("_no_pp.pkl", ""))

    print(f"{len(same_names)} / {len(no_tp_names)}")


if __name__ == "__main__":
    main()
