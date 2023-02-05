import oslo
from oslo import ParallelMode
from oslo.torch.nn.parallel import TensorParallel, PipelineParallel


def initialize_oslo(args, model):
    try:
        pc = oslo.ParallelContext.from_torch(
            data_parallel_size=args.data_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            tensor_parallel_size=args.tensor_parallel_size,
            tensor_parallel_depth=args.tensor_parallel_depth,
            tensor_parallel_mode={
                "1D": ParallelMode.TENSOR_1D,
                "2D": ParallelMode.TENSOR_2D,
                "2P5D": ParallelMode.TENSOR_2P5D,
                "3D": ParallelMode.TENSOR_3D,
            }[args.tensor_parallel_mode],
        )

        if pc.get_world_size(ParallelMode.TENSOR) > 1:
            model = TensorParallel(model, pc)
        if pc.get_world_size(ParallelMode.PIPELINE) > 1:
            model = PipelineParallel(model, pc)

        oslo.ready(model, pc)

    except:
        pc = None
        model = model.cuda()

    return model, pc


def print_rank_0(message, pc):
    if pc is None:
        print(message)
    elif pc.get_global_rank() == 0:
        print(message)
