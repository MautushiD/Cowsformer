import os
import datasets
import numpy as np
import pandas as pd


class Cow50(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="Cow50",
            version=VERSION,
            description="50+ cow standing images for segmentation",
        )
    ]

    def _info(self):
        features = datasets.Features(
            {
                "filename": datasets.Value("string"),
                "count": datasets.Value("int32"),
                "image": datasets.Image(),
                "annotation": datasets.Image(),
            }
        )
        return datasets.DatasetInfo(
            features=features,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data": ".",
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, data, split):
        WD = self.base_path
        dir_images = os.path.join(WD, split, "images")
        dir_annotations = os.path.join(WD, split, "annotations")

        ## file-driven solution
        # dir_images = os.path.join(WD, split, "images")
        # dir_annotations = os.path.join(WD, split, "annotations")
        # for idx, image_file in enumerate(os.listdir(dir_images)):
        #     image_id = image_file.split(".")[0]
        #     yield idx, {
        #         "image": os.path.join(dir_images, image_file),
        #         "annotation": os.path.join(dir_annotations, image_id + ".jpeg"),
        #     }
        # table-driven solution
        df = pd.read_csv(os.path.join(WD, split, "annotations.csv"))
        for idx, row in df.iterrows():
            filename = row["file"]
            count = row["n"]
            yield idx, {
                "filename": filename,
                "image": os.path.join(dir_images, filename),
                "annotation": os.path.join(dir_annotations, filename),
                "count": count,
            }

