# Refocus
Post-captured Dynamic Focus & Aperture Control

This repo contains code to run **Refocus** - an image editing technique to change focus point and f-stop setting in a post-processing manner.

![](/assets/title.png)

## Run Refocus

To reproduce our work, please do the following steps:
1. Clone the repo
    ```
    git clone https://github.com/KhanhNgoDuy/Refocus.git
    ```
2. Prepare your python environment
3. Install necessary packages
    ```
    pip install -r requirements.txt
    ```
4. Run
    ```
    python main.py --img_path <path/to/your/test_img>
    ```
    If the path is not specified, the code defaults to test on our prepared example.

## Current results & UI
![](/assets/windows.png)

## Methodology
Our solution implements two distinct approaches to address the refocusing task, a physics-based model and a deep learning-based model.


### Physics-Based Approach
The physics-based approach leverages the canonical principle in photography: Circle of Confusion.

![](/assets/physical_pipeline.png)

### Deep Learning-Based Approach
Our deep learning-based approach leverages Diffusion Models to learn complex real lens optical effects, which are difficult to model explicitly through physical equations.

**Training** \
Below is our training pipeline, training data is available at [NTIRE 2023 Bokeh Effect Transformation](https://codalab.lisn.upsaclay.fr/competitions/10229).

![](/assets/dm_pipeline_train.png)

**Testing** \
Our test data is available [here](https://drive.google.com/drive/folders/1BVcK7xJVfLbyX7gItURk33Yo419yBGVa?usp=sharing).
