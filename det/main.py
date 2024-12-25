from det.base import BasePipeline
from cfg.pipeline import detection_tracking, viz, diplaysink, filesink, segment, sm
from cfg.src import source

if __name__ == "__main__":
    str_pipeline = {**source, **sm, **detection_tracking, **viz, **diplaysink}
    # print(str_pipeline)
    pipeline = BasePipeline()
    pipeline.run(str_pipeline)