# wishart-textures

Examine noise correlations in response to texture stimuli.

## Installation

All requirements are listed in the included requirements file, and can be
installed (ideally, in a clean virtual environment) with:

``` sh
pip -r requirements.txt
```

This should work with python version at or above 3.9

Currently, this points to a specific branch of plenoptic because I'm doing some
work on the Portilla-Simoncelli texture model. I'll update the requirements file
and the relevant code as that progresses. It will speed the code up, but
shouldn't change the outputs.

## Usage

Currently, the main use of this repo is to generate "mixed texture metamers".
That is, we run the Portilla-Simoncelli texture model on $k$ grayscale images,
calculate a weighted average of their representations, and then synthesize a
metamer that matches that mixed representation.

That is, let $M$ be the Portilla-Simoncelli texture model, $x_i$ be natural
texture image $i$, $M(x_i)$ be the texture representation of $x_i$ (a vector
containing about 700 values) and $\alpha_i$ be the relative weight of that
image. We ensure that $\alpha_i > 0\ \ \forall i$ and $\sum_i \alpha_i = 1$, and
synthesize a metamer $\hat{x}$, which satisfies $argmin_{\hat{x}} (\sum_i\alpha_i
M(x_{i}) - M(\hat{x}))^2$.

To synthesize these textures, call

``` sh
python -m generate_textures.main CONFIG OUTPUT_DIR
```

where `CONFIG` is a yml file like the `config.yml` included in this repo (see
comments in that file for explanation of the different arguments), and
`OUTPUT_DIR` is the directory to put our output directory in. View `python -m
generate_textures.main --help` to get an explanation of how the output will be
structured. 
