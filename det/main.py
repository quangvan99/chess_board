from base import BasePipeline
from cfg import source, detection_tracking, viz, diplaysink, filesink, segment

if __name__ == "__main__":
    str_pipeline = {**source, **detection_tracking, **viz, **filesink}
    # print(str_pipeline)
    pipeline = BasePipeline()
    pipeline.run(str_pipeline)