"""Study Two on the author's augmented, class-imbalanced cylinder dataset."""

from mobilenetv2_study_one import AM_URL, REVISION, Study, main


STUDY = Study(
    name="two",
    classes=("Badaug", "Goodaug"),
    counts=(947, 2070),
    split_sizes=(2111, 308, 598),
    member_prefix=f"AMdataset-{REVISION}/Augmented_data/",
    inventory_sha256="737836bc23fcb0ce410a0207d1ee34653a24dd5b16724bfc7c7a22a5ab0fccd9",
    url=AM_URL,
    archive_name="amdataset.zip",
    epochs=50,
    batch_size=50,
    # Test rows of StudyTwo's MobileNetV2 workbooks, in the shared SCORES order.
    paper={
        "adam": (594 / 598, 403 / 407, 1.0, 806 / 810, 0.9848150942490522,
                 0.9846998170630301, 0.9974295348985176, 0.9975186256626635, 191 / 195),
        "sgd": (594 / 598, 403 / 407, 1.0, 806 / 810, 0.9848150942490522,
                0.9846998170630301, 0.9990201692434942, 0.9995168269334922, 191 / 195),
        "rmsprop": (597 / 598, 403 / 404, 1.0, 806 / 807, 0.996197390542704,
                    0.9961901606758323, 0.9999872749252402, 0.9999938579465887, 194 / 195),
    },
)


if __name__ == "__main__":
    main(STUDY)
