# Contributing

Contributions are welcome and greatly appreciated!
Please ensure that all contributions adhere to the [code of conduct](code_of_conduct.md).

## Types of Contributions

### Report Bugs

Report bugs on the `popkinmocks` [GitHub issues page](https://github.com/prashjet/popkinmocks/issues). Please report bugs including:

* Your operating system name and version.
* Any details about your Python environment.
* Detailed steps to reproduce the bug.

### Propose New Features

The best way to send feedback is to create an issue on the `popkinmocks` [GitHub issues page](https://github.com/prashjet/popkinmocks/issues) with tag _enhancement_.

If you are proposing a new feature:

* Explain in detail how it should work.
* Keep the scope as narrow as possible, to make it easier to implement.

### Implement Features
Look through the Git issues for feature requests.
Anything tagged with *enhancement* is open to whoever wants to
implement it.

### Add Examples or improve Documentation
Writing new features is not the only way to get involved and
contribute. Create examples with existing features as well 
as improving the documentation very much encouraged.

## Getting Started to contribute

Ready to contribute?

1. Follow the [installation instructions](install.rst).

2. Create a feature branch for local development:
    ```
    git checkout -b feature/name-of-your-branch
    ```
    Now you can make your changes locally.

3. When you're done making changes, run the tests (old and new) from the main directory and check that they pass successfully:
    ```
    cd popkinmocks/
    pytest
    ```

4. Install the [Black][black] code formatter and run it from the main directory:
    ```
    cd popkinmocks/
    black .
    ```

5. Commit your changes and push your branch to GitHub::
    ```
    git add .
    git commit -m "Your detailed description of your changes."
    git push origin feature/name-of-your-branch
    ```
    Remember to add ``-u`` when pushing the branch for the first time.

6. Submit a pull request through the GitHub website.


### Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include new tests for all the core routines that have been developed.
2. If the pull request adds functionality, the docs should be updated accordingly.

## Attribution

This code of conduct has been adapted from the [PyAutoLens][pyautolens] project.

[black]: https://black.readthedocs.io/en/stable/
[pyautolens]: https://github.com/Jammy2211/PyAutoLens