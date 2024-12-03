# Contributing to SwanLab

Interested in contributing to SwanLab? We welcome contributions from the community! This guide discusses the development workflow and internal structure of `swanlab`.

## 📦 Table of Contents

- [Standard Development Process](#standard-development-process)
- [Local Debugging](#local-debugging)
  - [IDE and Plugins](#ide-and-plugins)
  - [Configuring Python Environment](#configuring-python-environment)
  - [Debugging Scripts](#debugging-scripts)
- [Local Testing](#local-testing)
  - [Python Script Debugging](#python-script-debugging)
  - [Unit Testing](#unit-testing)

## Standard Development Process

1. Browse the [Issues](https://github.com/SwanHubX/SwanLab/issues) on GitHub to see the features you'd like to add or bugs you'd like to fix, and whether they have already been addressed by a Pull Request.

    - If not, create a [new Issue](https://github.com/SwanHubX/SwanLab/issues/new/choose) — this will help the project track feature requests and bug reports and ensure that work is not duplicated.

2. If you are contributing to an open-source project for the first time, go to the [project homepage](https://github.com/SwanHubX/SwanLab) and click the "Fork" button in the upper right corner. This will create a personal copy of the repository for your development.

    - Clone the forked project to your computer and add a remote link to the `swanlab` project:

   ```bash
   git clone https://github.com/<your-username>/swanlab.git
   cd swanlab
   git remote add upstream https://github.com/swanhubx/swanlab.git
   ```

3. Develop your contribution

    - Ensure your fork is synchronized with the main repository:

   ```bash
   git checkout main
   git pull upstream main
   ```

    - Create a `git` branch where you will develop your contribution. Use a reasonable name for the branch, for example:

   ```bash
   git checkout -b <username>/<short-dash-seperated-feature-description>
   ```

    - As you make progress, commit your changes locally, for example:

   ```bash
   git add changed-file.py tests/test-changed-file.py
   git commit -m "feat(integrations): Add integration with the `awesomepyml` library"
   ```

4. Submit your contribution:

    - [Github Pull Request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests)
    - When your contribution is ready, push your branch to GitHub:

   ```bash
   git push origin <username>/<short-dash-seperated-feature-description>
   ```

    - After the branch is uploaded, `GitHub` will print a URL to submit your contribution as a pull request. Open that URL in your browser, write an informative title and detailed description for your pull request, and submit it.

    - Please link the relevant Issue (existing Issue or the one you created) to your PR. See the right sidebar of the PR page. Alternatively, mention "Fixes issue link" in the PR description — GitHub will automatically link it.

    - We will review your contribution and provide feedback. To merge the changes suggested by the reviewer, commit the edits to your branch and push it again (no need to recreate the pull request, it will automatically track modifications to the branch), for example:

   ```bash
   git add tests/test-changed-file.py
   git commit -m "test(sdk): Add a test case to address reviewer feedback"
   git push origin <username>/<short-dash-seperated-feature-description>
   ```

    - Once your pull request is approved by the reviewer, it will be merged into the main branch of the repository.

## Local Debugging

### IDE and Plugins

1. **Use VSCode as your development IDE**

   The SwanLab repository has already configured the [VSCode](https://code.visualstudio.com/) environment, plugins, and debugging scripts (located in the `.vscode` folder). Developing SwanLab with VSCode will provide the best experience.

2. **Install VSCode Plugins (Optional)**

   Open the project with VSCode, go to [Extensions], enter "@recommended" in the search box, and a series of recommended plugins will appear. It is recommended to install all these plugins.

   ![vscode-recommend](/assets/guide_cloud/community/contributing-code/vscode_recommend.png)

### Configuring Python Environment

The SwanLab project environment requires `python>=3.8`.

The necessary Python dependencies are centrally recorded in `requirements.txt` in the project root directory.

Start the terminal in the project root directory and run the following command to install the dependencies:

```Bash
# Packages required by swanlab
pip install -r requirements.txt
pip install -r requirements-media.txt
```

Additional dependencies are required for compilation, development, unit testing, etc.:

```Bash
# Packages required for compilation, unit testing, etc.
pip install -r requirements-dev.txt
```

### Debugging Scripts

1. **VSCode Debugging Scripts**

In VSCode - Run and Debug, the project has configured a series of debugging scripts:

![img](/assets/guide_cloud/community/contributing-code/debug.png)

- **Start an Experiment**: Run the `test/create_experiment.py` script

- **Run the Current File**: Use the configured Python environment to run the file you selected

- **Test the Current File**: Test the file you selected in debug mode

- **Run All Unit Tests**: Run the scripts in `test/unit` to perform a complete unit test of the basic functions of swanlab

- **(Skip Cloud) Run All Unit Tests**: Run the scripts in `test/unit` to perform a complete unit test of the basic functions of swanlab, but skip cloud testing

- **Build the Project**: Package the project into a whl file (pip installation package format)

Ps: If you do not want to use VSCode for development, you can go to `.vscode/launch.json` to view the command corresponding to each debugging item to understand its configuration.

## Local Testing

The prerequisite for testing is that you have installed all the required dependencies.

### Python Script Debugging

After completing your changes, you can place your Python test script in the root directory or the `test` folder, and then use the "Run the Current File" in [VSCode Scripts](#debugging-scripts) to run your Python test script. This way, your script will use the modified swanlab.

### Unit Testing

You can perform unit testing through [VSCode Scripts](#debugging-scripts) or by running the following command in the project root directory:

```Bash
export PYTHONPATH=. && pytest test/unit
```

Since swanlab involves interaction with the cloud, and the cloud part is closed-source, the simplest way if you are contributing code for the first time is to only perform local testing.
For this situation, create a `.env` file in the local root directory and fill in the following environment variable configuration:

```dotenv
SWANLAB_RUNTIME=test-no-cloud
```

This will skip cloud testing and only perform local part function testing. If you want to perform complete testing, please supplement the following information in `.env`:

```dotenv
SWANLAB_RUNTIME=test
SWANLAB_API_KEY=<your API KEY>
SWANLAB_API_HOST=https://swanlab.cn/api
SWANLAB_WEB_HOST=https://swanlab.cn
```

*Note: When performing cloud testing, some useless test experiment data will be generated under your cloud account, which needs to be manually deleted*

After configuring, you can run the complete test