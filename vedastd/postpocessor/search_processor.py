from .registry import POSTPROCESS
from .builder import build_postprocessor


@POSTPROCESS.register_module
class SearchPostprocessor:

    def __init__(self, post_processors: list):
        self.cfgs = post_processors
        self.postprocessors = [build_postprocessor(cfg)
                               for cfg in post_processors]

    def __len__(self):
        return len(self.postprocessors)

    def __call__(self, batch, _pred, training=False):
        return [postprocessor(batch, _pred, training)
                for postprocessor in self.postprocessors]
