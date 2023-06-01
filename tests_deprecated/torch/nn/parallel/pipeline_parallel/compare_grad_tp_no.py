import os
import glob

import torch
import einops


save_dir = "tmp_tp"


def main():
    file_names = os.listdir(f"{save_dir}/")
    no_tp_names = sorted([fn for fn in file_names if "no_tp" in fn and "grad_" in fn])

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

        print(f"{tp_data[0].shape=}, {no_tp_data.shape=}")

        if len(no_tp_data.shape) == 1:  # bias
            if no_tp_data.size(0) == tp_data[0].size(0):
                tp_data = tp_data[0]
            else:
                if any([x in no_tp_name for x in [".c_attn"]]):
                    for i in range(len(tp_data)):
                        tp_data[i] = einops.rearrange(tp_data[i], "(n d) -> n d", n=3)
                    tp_data = torch.stack(tp_data, 0)
                    tp_data = einops.rearrange(tp_data, "m n d -> (n m d)")
                else:
                    tp_data = torch.cat(tp_data, 0)
        else:
            # column split + qkv reshape
            if any([x in no_tp_name for x in [".c_attn"]]):
                n = len(tp_data)
                for i in range(n):
                    tp_data[i] = torch.transpose(tp_data[i], 1, 0)
                    tp_data[i] = einops.rearrange(tp_data[i], "o (n d) -> o n d", n=3)
                tp_data = torch.stack(tp_data, 0)
                tp_data = einops.rearrange(tp_data, "m o n d -> o (n m d)")

            # column split
            elif any([x in no_tp_name for x in [".c_fc"]]):
                for i in range(len(tp_data)):
                    tp_data[i] = torch.transpose(tp_data[i], 1, 0)
                tp_data = torch.cat(tp_data, -1)

            # row split
            elif any([x in no_tp_name for x in [".c_proj"]]):
                for i in range(len(tp_data)):
                    tp_data[i] = torch.transpose(tp_data[i], 1, 0)
                tp_data = torch.cat(tp_data, 0)

            # wpe
            elif any([x in no_tp_name for x in [".wpe"]]):
                tp_data = torch.cat(tp_data, -1)

            # wte
            elif any([x in no_tp_name for x in [".wte"]]):
                tp_data = torch.cat(tp_data, 0)

        if any([x in no_tp_name for x in [".wpe", ".wte"]]):
            tp_data = tp_data[:no_tp_data.size(0)]

        print(f"{tp_data.shape=}, {no_tp_data.shape=}")

        if not torch.allclose(tp_data, no_tp_data):
            print(f"   >>> diff: {torch.sum(torch.abs(tp_data - no_tp_data))}")
            # print(pp_name)

            diff_cnt += 1
            diff_names.append(no_tp_name)

            # break
        else:
            same_names.append(no_tp_name)

    print("Names with different gradient: ")
    for dn in diff_names:
        print(dn.replace("_no_pp.pkl", ""))

    print(f"{diff_cnt} / {len(no_tp_names)}")

    print("Names with same gradient: ")
    for sn in same_names:
        print(sn.replace("_no_pp.pkl", ""))

    print(f"{len(same_names)} / {len(no_tp_names)}")


if __name__ == "__main__":
    main()
