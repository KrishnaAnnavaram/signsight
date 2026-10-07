# data/

Git ignores all files in this folder except this README. Do not commit images, archives or models.

## 1. Synthetic signs (default, no download)

`signsight.synthetic.make_signs` draws 10 sign classes (shape + colour + inner mark) in
tracks: each track is one sign instance with its own background, colour shade and position,
and its frames show the sign while it grows. The demo and the tests make them in memory.
`signsight make-scenes --out data/scenes` writes larger street-like images with one sign
each, for `predict` and `latency`. These images are not real traffic signs.

## 2. GTSRB (real data)

| Item | Value |
|---|---|
| Source | German Traffic Sign Recognition Benchmark, Institut für Neuroinformatik, Ruhr-Universität Bochum |
| URL | https://benchmark.ini.rub.de/gtsrb_dataset.html |
| Terms | Free for research. Read the terms on the dataset page |
| Training file | `GTSRB_Final_Training_Images.zip` (39,209 images, 43 classes) |
| Official test files | `GTSRB_Final_Test_Images.zip` and `GTSRB_Final_Test_GT.zip` (12,630 images) |
| Annotation | `GT-<class>.csv`: `Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId` |
| Tracks | File name `<track>_<frame>.ppm`, about 30 frames for each physical sign |

```bash
signsight benchmark --data data/GTSRB_Final_Training_Images.zip \
    --test-images data/GTSRB/Final_Test/Images --test-gt data/GT-final_test.csv
```

The reader reads the zip directly, crops each image to its region of interest and resizes
it to `SIGNSIGHT_IMAGE_SIZE`. The validation part is split by track. The official test set
is used only for the final numbers.

## 3. Your own photos

Put photos in a folder and run `signsight predict --model runs/model.joblib --images <folder>`.
The colour detector crops the sign first. Photos from another country or camera are a
different domain: check the `unknown` answers and the confidence values.
