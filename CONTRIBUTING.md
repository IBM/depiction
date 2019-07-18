## Contributing In General

Our project welcomes external contributions. If you have an itch, please feel
free to scratch it.

To contribute code or documentation, please submit a [pull request](https://github.com/ibm/dl-interpretability-compbio/pulls).

A good way to familiarize yourself with the codebase and contribution process is
to look for and tackle low-hanging fruit in the [issue tracker](https://github.com/ibm/dl-interpretability-compbio/issues).
Before embarking on a more ambitious contribution, please quickly [get in touch](#communication) with us.

**Note: We appreciate your effort, and want to avoid a situation where a contribution
requires extensive rework (by you or by us), sits in backlog for a long time, or
cannot be accepted at all!**

### Proposing new features

If you would like to implement a new feature, please [raise an issue](https://github.com/ibm/dl-interpretability-compbio/issues)
before sending a pull request so the feature can be discussed. This is to avoid
you wasting your valuable time working on a feature that the project developers
are not interested in accepting into the code base.

### Fixing bugs

If you would like to fix a bug, please [raise an issue](https://github.com/ibm/dl-interpretability-compbio/issues) before sending a
pull request so it can be tracked.

### Merge approval

The project maintainers use LGTM (Looks Good To Me) in comments on the code
review to indicate acceptance. A change requires LGTMs from two of the
maintainers of each component affected.

For a list of the maintainers, see the [MAINTAINERS.md](MAINTAINERS.md) page.

## Legal

We have tried to make it as easy as possible to make contributions. This
applies to how we handle the legal aspects of contribution. We use the
same approach - the [Developer's Certificate of Origin 1.1 (DCO)](https://github.com/hyperledger/fabric/blob/master/docs/source/DCO1.1.txt) - that the Linux® Kernel [community](https://elinux.org/Developer_Certificate_Of_Origin)
uses to manage code contributions.

We simply ask that when submitting a patch for review, the developer
must include a sign-off statement in the commit message.

Here is an example Signed-off-by line, which indicates that the
submitter accepts the DCO:

```console
Signed-off-by: Jane Doe <jane.doe@example.com>
```

You can include this automatically when you commit a change to your
local git repository using the following command:

```sh
git commit -s
```

## Communication

Please feel free to connect with us via email, see the [MAINTAINERS.md](MAINTAINERS.md) page.

## Setup

Setup a conda environment

```sh
conda env create -f environment.yml
```

Activate it:

```sh
conda activate depiction-env
```

Install the module in editable mode:

```sh
pip install -e .
```

Optionally, install a `jupiter` playground:

```sh
pip install jupyter
ipython kernel install --user --name=depiction-development
```

## Testing

For tests we use the `unittest` module.
You can run tess by typing:

```sh
python -m unittest discover -p "*test.py"
```

## Coding style guidelines

We try to follow PEP8 styling.
