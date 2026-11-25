"""Study Three on the public Kaggle 3D-Printer Defected Dataset."""

from mobilenetv2_study_one import Study, main


STUDY = Study(
    name="three",
    classes=("defected", "no_defected"),
    counts=(759, 798),
    split_sizes=(1089, 159, 309),
    member_prefix="",
    inventory_sha256="01acdbbf2c6debba50ded98d5c3ea6e3f5ddb539dc3b0236c42a0f05a0a94017",
    url=("https://www.kaggle.com/api/v1/datasets/download/"
         "justin900429/3d-printer-defected-dataset?datasetVersionNumber=1"),
    archive_name="study-three-v1.zip",
    epochs=30,
    batch_size=30,
    # Test rows of StudyThree's MobileNetV2 workbooks, in the shared SCORES order.
    paper={
        "adam": (307 / 309, 156 / 158, 1.0, 312 / 314, 0.9871349166946514,
                 0.9870521684475173, 0.9997905144963969, 0.9997958417034967, 151 / 153),
        "sgd": (294 / 309, 148 / 155, 148 / 156, 296 / 311, 0.9029284854377749,
                0.9029095707911771, 0.9943857885034355, 0.9944626530735788, 146 / 153),
        "rmsprop": (307 / 309, 156 / 158, 1.0, 312 / 314, 0.9871349166946514,
                    0.9870521684475173, 0.9997486173956762, 0.9997550154219756, 151 / 153),
    },
)


if __name__ == "__main__":
    main(STUDY)
