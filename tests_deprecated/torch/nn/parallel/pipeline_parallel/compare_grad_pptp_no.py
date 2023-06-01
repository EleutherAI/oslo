import os
import glob

import torch
import einops


save_dir = "tmp3"


def main():
    file_names = os.listdir(f"{save_dir}/")
    no_pp_names = sorted([fn for fn in file_names if "no_pp" in fn and "grad_" in fn])

    print(len(no_pp_names))

    diff_cnt = 0
    diff_names = []
    same_names = []
    for no_pp_name in no_pp_names:
        pp_tp_name_template = no_pp_name.split("no_pp")[0]
        pp_tp_name_template = pp_tp_name_template + "pp_tp_*.pkl"
        pp_tp_paths = glob.glob(os.path.join(f"{save_dir}", pp_tp_name_template))
        pp_tp_paths = sorted(pp_tp_paths)

        print()
        print(pp_tp_name_template)
        print(pp_tp_paths)

        no_pp_path = os.path.join(f"{save_dir}", no_pp_name)
        no_pp_data = torch.load(no_pp_path, map_location="cpu")

        pp_tp_data = []
        for path in pp_tp_paths:
            data = torch.load(path, map_location="cpu")
            pp_tp_data.append(data)

        print(f"{pp_tp_data[0].shape=}, {no_pp_data.shape=}")

        if len(no_pp_data.shape) == 1:  # bias
            if no_pp_data.size(0) == pp_tp_data[0].size(0):
                pp_tp_data = pp_tp_data[0]
            else:
                if any([x in no_pp_name for x in [".c_attn"]]):
                    for i in range(len(pp_tp_data)):
                        pp_tp_data[i] = einops.rearrange(pp_tp_data[i], "(n d) -> n d", n=3)
                    pp_tp_data = torch.stack(pp_tp_data, 0)
                    pp_tp_data = einops.rearrange(pp_tp_data, "m n d -> (n m d)")
                else:
                    pp_tp_data = torch.cat(pp_tp_data, 0)
        else:
            # column split + qkv reshape
            if any([x in no_pp_name for x in [".c_attn"]]):
                n = len(pp_tp_data)
                for i in range(n):
                    pp_tp_data[i] = torch.transpose(pp_tp_data[i], 1, 0)
                    pp_tp_data[i] = einops.rearrange(pp_tp_data[i], "o (n d) -> o n d", n=3)
                pp_tp_data = torch.stack(pp_tp_data, 0)
                pp_tp_data = einops.rearrange(pp_tp_data, "m o n d -> o (n m d)")

            # column split
            elif any([x in no_pp_name for x in [".c_fc"]]):
                for i in range(len(pp_tp_data)):
                    pp_tp_data[i] = torch.transpose(pp_tp_data[i], 1, 0)
                pp_tp_data = torch.cat(pp_tp_data, -1)

            # row split
            elif any([x in no_pp_name for x in [".c_proj"]]):
                for i in range(len(pp_tp_data)):
                    pp_tp_data[i] = torch.transpose(pp_tp_data[i], 1, 0)
                pp_tp_data = torch.cat(pp_tp_data, 0)

            # wpe
            elif any([x in no_pp_name for x in [".wpe"]]):
                pp_tp_data = torch.cat(pp_tp_data, -1)

            # wte
            elif any([x in no_pp_name for x in [".wte"]]):
                pp_tp_data = torch.cat(pp_tp_data, 0)

        if any([x in no_pp_name for x in [".wpe", ".wte"]]):
            pp_tp_data = pp_tp_data[:no_pp_data.size(0)]

        print(f"{pp_tp_data.shape=}, {no_pp_data.shape=}")

        if not torch.allclose(pp_tp_data, no_pp_data):
            print(f"   >>> diff: {torch.sum(torch.abs(pp_tp_data - no_pp_data))}")
            # print(pp_name)

            diff_cnt += 1
            diff_names.append(no_pp_name)

            # break
        else:
            same_names.append(no_pp_name)

    print("Names with different gradient: ")
    for dn in diff_names:
        print(dn.replace("_no_pp.pkl", ""))

    print(f"{diff_cnt} / {len(no_pp_names)}")

    print("Names with same gradient: ")
    for sn in same_names:
        print(sn.replace("_no_pp.pkl", ""))

    print(f"{len(same_names)} / {len(no_pp_names)}")


if __name__ == "__main__":
    main()
